use aya::{
    maps::XskMap,
    programs::{Xdp, XdpFlags},
};
use log::{error, info};

use std::{
    ffi::CStr,
    io::Error as ioError,
    ops::{Deref, DerefMut},
    os::fd::{AsRawFd, RawFd},
    ptr::NonNull,
    sync::atomic::{AtomicU32, Ordering},
};

use libc::{xdp_mmap_offsets, xdp_ring_offset};
use log::debug;
use thiserror::Error;

/// A Simple wrapper around a [NonNull] pointer to `T` that implements [Deref] and [DerefMut]
/// Invariant: The pointer must be convertible to a reference as specified by [NonNull::as_ref] and
/// [NonNull::as_mut]
#[repr(transparent)]
struct DerefNonNull<T>(NonNull<T>);

impl<T> DerefNonNull<T> {
    /// Returns a wrapped [NonNull] pointer
    ///
    /// # Safety
    /// When calling this method you must ensure that the pointer is convertible to a reference per
    /// the requirements specified by [NonNull] in the documentation for [NonNull::as_ref] and
    /// [NonNull::as_mut]
    unsafe fn new(ptr: NonNull<T>) -> Self {
        Self(ptr)
    }
}

impl<T> Deref for DerefNonNull<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        // Safety:
        // The invariant of the struct is that the pointer is convertible to a reference
        unsafe { self.0.as_ref() }
    }
}

impl<T> DerefMut for DerefNonNull<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        // Safety:
        // The invariant of the struct is that the pointer is convertible to a reference
        unsafe { self.0.as_mut() }
    }
}

#[derive(Error, Debug)]
pub enum UmemError {
    #[error("Failed to allocate umem buffer: {0}")]
    Alloc(ioError),
    #[error("Failed to register umem buffer: {0}")]
    Register(ioError),
    #[error("Fill buffer operation failure: {0}")]
    FillBuffer(RingBufferError),
    #[error("Completion buffer operation failure: {0}")]
    CompletionBuffer(RingBufferError),
}

/// A safe wrapper around a page-aligned UMEM buffer along with its associated fill and completion
/// buffers.
struct Umem<const COUNT: usize, const LEN: usize, const FILL: usize, const COMP: usize> {
    buffer: DerefNonNull<[u8; LEN]>,
    fill: FillBuffer<FILL>,
    comp: CompletionBuffer<COMP>,
}

impl<const COUNT: usize, const LEN: usize, const FILL: usize, const COMP: usize>
    Umem<COUNT, LEN, FILL, COMP>
{
    const CHUNK_SIZE: usize = LEN / COUNT;

    /// Allocates a page aligned buffer of size [Self::BUF_LEN]
    pub fn new(fd: &XdpFd, offsets: xdp_mmap_offsets) -> Result<Self, UmemError> {
        const {
            assert!(COUNT.is_power_of_two(), "Umem chunk count must be a power of 2");
            assert!(LEN.is_power_of_two(), "Umem chunk length must be a power of 2");
            assert!(LEN & COUNT == 0, "Umem chunk count must divide buffer size");
            assert!(Self::CHUNK_SIZE.is_power_of_two(), "Umem chunk size must be a power of 2");
        }

        let addr = unsafe {
            let addr = libc::mmap(
                std::ptr::null_mut(),
                LEN,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_ANONYMOUS | libc::MAP_PRIVATE,
                -1,
                0,
            );

            if addr == libc::MAP_FAILED {
                None
            } else {
                // mmap only allocates at null if MAP_FIXED is passed in flags
                Some(NonNull::new(addr).expect("NonNull new failed somehow"))
            }
        };
        let addr = addr.ok_or_else(|| UmemError::Alloc(ioError::last_os_error()))?;

        let buffer = unsafe { DerefNonNull::new(addr.cast()) };
        Self::register_buffer(&buffer, fd).map_err(UmemError::Register)?;

        let fill = FillBuffer::new(fd, offsets).map_err(UmemError::FillBuffer)?;
        let comp = CompletionBuffer::new(fd, offsets).map_err(UmemError::CompletionBuffer)?;

        Ok(Umem {
            buffer,
            fill,
            comp,
        })
    }

    /// Registers the [Umem] buffer with the specified [XdpSock]
    fn register_buffer(buf: &DerefNonNull<[u8; LEN]>, fd: &XdpFd) -> Result<(), ioError> {
        let umem_reg = libc::xdp_umem_reg {
            addr: buf.as_ptr() as _,
            len: LEN as _,
            chunk_size: Self::CHUNK_SIZE as _,
            headroom: 0,
            flags: 0,
            tx_metadata_len: 0,
        };

        // Safety: XDP_UMEM_REG is a valid operation and umem_reg is a valid parameter to pass it
        unsafe { setsockopt(&fd, libc::XDP_UMEM_REG, &umem_reg) }
    }
}

impl<const COUNT: usize, const LEN: usize, const FILL: usize, const COMP: usize> Drop for
    Umem<COUNT, LEN, FILL, COMP>
{
    fn drop(&mut self) {
        // Safety:
        // We acquired this memory through mmap and therefor its safe to pass it to munmap to free
        // it. The allocation always has the size Self::BUF_LEN
        unsafe {
            libc::munmap(self.buffer.as_mut_ptr().cast(), LEN);
        }
    }
}

#[derive(Debug, Error)]
pub enum RingBufferError {
    #[error("Failed to register ring size: {0}")]
    RegisterSize(ioError),
    #[error("Failed to allocate the buffer: {0}")]
    Alloc(ioError),
}

/// A generic safe-ish wrapper around an XDP ring buffer
struct RingBuffer<T, const N: usize> {
    addr: NonNull<libc::c_void>,
    mem_len: usize,
    producer: DerefNonNull<AtomicU32>,
    consumer: DerefNonNull<AtomicU32>,
    flags: DerefNonNull<AtomicU32>,
    data: DerefNonNull<[T; N]>,
    cached_prod: u32,
    cached_cons: u32,
}

impl<T, const N: usize> RingBuffer<T, N> {
    /// Allocates a ring buffer given an XDP socket and the offsets associated with it
    ///
    /// # Safety
    /// The caller must ensure that:
    /// * `offset_fn` returns the correct offset associated with the ring buffer type
    /// * `sockopt` is a valid size registration [setsockopt] operation name for registering ring
    ///   size with the kernel (e.g. [libc::XDP_RX_RING]) for this ring type and matches the ring
    ///   type specified by the value which `offset_fn` returns
    /// * `mmap_off` is a valid offset to pass to [libc::mmap] when allocating a ring buffer
    ///   (e.g. [libc::XDP_PGOFF_RX_RING]) for this ring type and matches the ring type specified
    ///   by `sockopt`
    unsafe fn new(
        fd: &XdpFd,
        offset: xdp_ring_offset,
        sockopt: libc::c_int,
        mmap_off: libc::off_t,
    ) -> Result<Self, RingBufferError> {
        const {
            assert!(N > 1, "Buffer size must be greater than 1");
            assert!(N.is_power_of_two(), "Buffer size must be a power of two")
        }

        // Safety:
        // Ring buffer length is always a power of two, invariant ensured by constructor
        // sockopt is a valid size registration op name as required by function contract
        unsafe { setsockopt(fd, sockopt, &N) }.map_err(RingBufferError::RegisterSize)?;

        let mem_len = offset.desc as usize + N * size_of::<T>();

        // Safety:
        // These are the correct parameters to pass to mmap when allocating a ring buffer and
        // mmap_off is a valid offset as required by function contract
        let addr = unsafe {
            let addr = libc::mmap(
                std::ptr::null_mut(),
                mem_len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED | libc::MAP_POPULATE,
                fd.as_raw_fd(),
                mmap_off,
            );

            if addr == libc::MAP_FAILED {
                None
            } else {
                // mmap only allocates at null if MAP_FIXED is passed in flags
                Some(NonNull::new(addr).expect("NonNull new failed somehow"))
            }
        };

        let addr = addr.ok_or_else(|| RingBufferError::Alloc(ioError::last_os_error()))?;

        // Safety:
        // All add operations are safe as the offsets are provided by the kernel and we allocated
        // memory large enough for all of them
        // All DerefNonNull::new are safe as the pointers are convertible to a reference which is
        // ensured by the fact that we got the pointers from the kernel
        let (producer, consumer, flags, data) = unsafe {
            (
                DerefNonNull::<AtomicU32>::new(addr.add(offset.producer as _).cast()),
                DerefNonNull::<AtomicU32>::new(addr.add(offset.consumer as _).cast()),
                DerefNonNull::<AtomicU32>::new(addr.add(offset.flags as _).cast()),
                DerefNonNull::new(addr.add(offset.desc as _).cast()),
            )
        };

        Ok(Self {
            addr,
            mem_len,
            cached_prod: producer.load(Ordering::SeqCst),
            cached_cons: consumer.load(Ordering::SeqCst),
            producer,
            consumer,
            flags,
            data,
        })
    }
}

impl<T, const N: usize> Drop for RingBuffer<T, N> {
    fn drop(&mut self) {
        // Safety:
        // We acquired this memory through mmap and therefor its safe to pass it to munmap to free
        // it. The allocation size is always stored in mem_len
        unsafe {
            libc::munmap(self.addr.cast().as_ptr(), self.mem_len);
        }
    }
}

/// A safe wrapper around an XDP socket RX buffer
#[repr(transparent)]
struct RxBuffer<const N: usize>(RingBuffer<libc::xdp_desc, N>);
/// A safe wrapper around an XDP socket TX buffer
#[repr(transparent)]
struct TxBuffer<const N: usize>(RingBuffer<libc::xdp_desc, N>);
/// A safe wrapper around an XDP socket fill buffer
#[repr(transparent)]
struct FillBuffer<const N: usize>(RingBuffer<u64, N>);
/// A safe wrapper around an XDP socket completion buffer
#[repr(transparent)]
struct CompletionBuffer<const N: usize>(RingBuffer<u64, N>);

impl<const N: usize> RxBuffer<N> {
    /// Constructs a new [RxBuffer] associated with the given fd
    ///
    /// # Returns
    /// An [RxBuffer] or a [RingBufferError] on error
    pub fn new(fd: &XdpFd, offsets: xdp_mmap_offsets) -> Result<Self, RingBufferError> {
        // Safety:
        // the arguments provided are the correct ones for constructing rx buffers
        unsafe { RingBuffer::new(fd, offsets.rx, libc::XDP_RX_RING, libc::XDP_PGOFF_RX_RING) }
            .map(Self)
    }
}

impl<const N: usize> Deref for RxBuffer<N> {
    type Target = RingBuffer<libc::xdp_desc, N>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<const N: usize> DerefMut for RxBuffer<N> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<const N: usize> TxBuffer<N> {
    /// Constructs a new [TxBuffer] associated with the given fd
    ///
    /// # Returns
    /// A [TxBuffer] or a [RingBufferError] on error
    pub fn new(fd: &XdpFd, offsets: xdp_mmap_offsets) -> Result<Self, RingBufferError> {
        // Safety:
        // the arguments provided are the correct ones for constructing tx buffers
        unsafe { RingBuffer::new(fd, offsets.tx, libc::XDP_TX_RING, libc::XDP_PGOFF_TX_RING) }
            .map(Self)
    }
}

impl<const N: usize> Deref for TxBuffer<N> {
    type Target = RingBuffer<libc::xdp_desc, N>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<const N: usize> DerefMut for TxBuffer<N> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<const N: usize> FillBuffer<N> {
    /// Constructs a new [FillBuffer] associated with the given fd
    ///
    /// # Returns
    /// A [FillBuffer] or a [RingBufferError] on error
    pub fn new(fd: &XdpFd, offsets: xdp_mmap_offsets) -> Result<Self, RingBufferError> {
        // Safety:
        // the arguments provided are the correct ones for constructing fill buffers
        unsafe {
            RingBuffer::new(
                fd,
                offsets.fr,
                libc::XDP_UMEM_FILL_RING,
                libc::XDP_UMEM_PGOFF_FILL_RING as _,
            )
        }
        .map(Self)
    }
}

impl<const N: usize> Deref for FillBuffer<N> {
    type Target = RingBuffer<u64, N>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<const N: usize> DerefMut for FillBuffer<N> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<const N: usize> CompletionBuffer<N> {
    /// Constructs a new [CompletionBuffer] associated with the given fd
    ///
    /// # Returns
    /// A [CompletionBuffer] or a [RingBufferError] on error
    pub fn new(fd: &XdpFd, offsets: xdp_mmap_offsets) -> Result<Self, RingBufferError> {
        // Safety:
        // the arguments provided are the correct ones for constructing completion buffers
        unsafe {
            RingBuffer::new(
                fd,
                offsets.cr,
                libc::XDP_UMEM_COMPLETION_RING,
                libc::XDP_UMEM_PGOFF_COMPLETION_RING as _,
            )
        }
        .map(Self)
    }
}

impl<const N: usize> Deref for CompletionBuffer<N> {
    type Target = RingBuffer<u64, N>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<const N: usize> DerefMut for CompletionBuffer<N> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

/// A thin low-level wrapper around an XDP socket file descriptor
pub struct XdpFd(libc::c_int);

impl XdpFd {
    /// Attempts to create a new [XdpFd]
    ///
    /// # Returns
    /// An [XdpFd] or [std::io::Error] on error
    fn new() -> Result<XdpFd, ioError> {
        // Safety:
        // This function returns -1 on failure to create a socket, and is not unsafe to call.
        // We check that the fd returned is valid before creating an XdpFd out of it which ensures
        // a valid struct.
        let fd = unsafe { libc::socket(libc::AF_XDP, libc::SOCK_RAW, 0) };
        if fd == -1 {
            return Err(ioError::last_os_error());
        }

        Ok(Self(fd))
    }
}

impl AsRawFd for XdpFd {
    fn as_raw_fd(&self) -> RawFd {
        self.0
    }
}

impl Drop for XdpFd {
    fn drop(&mut self) {
        // Safety:
        // This function is safe to call on a file descriptor so long as it is a valid one
        // returned from a socket or open call. We have a valid file descriptor as it is
        // is not possible to construct an instance of XdpFd without one.
        unsafe {
            libc::close(self.0);
        }
    }
}

#[derive(Debug)]
pub struct NetworkInterface {
    name: String,
    index: u32,
}

impl NetworkInterface {
    pub fn new(name: String, index: u32) -> Self {
        NetworkInterface {
            name,
            index,
        }
    }
}

#[derive(Debug, Error)]
pub enum XdpSockError {
    #[error("Failed to create xdp socket: {0}")]
    Socket(ioError),
    #[error("Failed to create umem buffer: {0}")]
    Umem(UmemError),
    #[error("Failed to get ring offsets: {0}")]
    RingOffsets(ioError),
    #[error("Rx buffer operation failed: {0}")]
    RxError(RingBufferError),
    #[error("Tx buffer operation failed: {0}")]
    TxError(RingBufferError),
    #[error("Failed to find an appropriate network interface")]
    IfiSearchFail,
    #[error("Failed to find an appropriate network interface due to error: {0}")]
    IfiSearchError(ioError),
    #[error("Failed to bind xdp socket to network interface: {0}")]
    BindError(ioError),
}

/// A ready-to-use high level wrapper around an XDP socket
pub struct XdpSock {
    /// The XDP socket file descriptor
    fd: XdpFd,
    /// The UMEM buffer, fill and completion buffers
    umem: Umem<{Self::UMEM_CHUNK_COUNT}, {Self::UMEM_LEN}, {Self::RING_LEN}, {Self::RING_LEN}>,
    /// The RX buffer
    rx: RxBuffer<{Self::RING_LEN}>,
    /// The TX buffer
    tx: TxBuffer<{Self::RING_LEN}>,
    /// NIC's info
    nic: NetworkInterface,
}

impl XdpSock {
    const UMEM_CHUNK_SIZE: usize = 4096;
    const UMEM_CHUNK_COUNT: usize = 4096;
    const UMEM_LEN: usize = Self::UMEM_CHUNK_SIZE * Self::UMEM_CHUNK_COUNT;
    const RING_LEN: usize = 1024;

    /// Attempts to create an [XdpSock]
    ///
    /// # Returns
    /// An [XdpSock] or an [XdpSockError] on error
    pub fn new() -> Result<XdpSock, XdpSockError> {
        let fd = XdpFd::new().map_err(XdpSockError::Socket)?;

        let offsets = get_ring_offsets(&fd).map_err(XdpSockError::RingOffsets)?;

        let umem = Umem::new(&fd, offsets).map_err(XdpSockError::Umem)?;

        let rx = RxBuffer::new(&fd, offsets).map_err(XdpSockError::RxError)?;
        let tx = TxBuffer::new(&fd, offsets).map_err(XdpSockError::TxError)?;

        let nic = Self::autodetect_nic()
            .map_err(XdpSockError::IfiSearchError)?
            .ok_or(XdpSockError::IfiSearchFail)?;

        debug!("selected NIC: {:?}", nic);

        Self::bind(&fd, nic.index).map_err(XdpSockError::BindError)?;

        Ok(Self {
            fd,
            umem,
            rx,
            tx,
            nic,
        })
    }

    /// Binds `fd` and its associated resources with the specified NIC by index.
    ///
    /// # Returns
    /// An empty [Ok] value to represent success, or an [std::io::Error] if bind failed.
    fn bind(fd: &XdpFd, ifindex: u32) -> Result<(), ioError> {
        let sockaddr = libc::sockaddr_xdp {
            sxdp_family: libc::AF_XDP as _,
            sxdp_flags: libc::XDP_COPY | libc::XDP_USE_NEED_WAKEUP,
            sxdp_ifindex: ifindex,
            sxdp_queue_id: 0,
            sxdp_shared_umem_fd: 0,
        };

        let ret = unsafe {
            libc::bind(
                fd.as_raw_fd(),
                &sockaddr as *const _ as *const _,
                size_of::<libc::sockaddr_xdp>() as _,
            )
        };

        if ret == 0 {
            Ok(())
        } else {
            Err(ioError::last_os_error())
        }
    }

    /// Attempts to detect which network interface to bind to using some simple heuristics. This
    /// function will return the first NIC it encounters that is:
    /// * running
    /// * not a loopback device
    /// * has ARP support/an L2 address
    ///
    /// # Return
    /// If a NIC meeting all the criteria is found then it will be returned, if not then the value
    /// `Ok(None)` will be returned, otherwise a system error will be returned.
    fn autodetect_nic() -> Result<Option<NetworkInterface>, ioError> {
        let mut ifaddrs: *mut libc::ifaddrs = std::ptr::null_mut();
        unsafe {
            libc::getifaddrs(&mut ifaddrs as *mut _);
        }

        if ifaddrs.is_null() {
            return Err(ioError::last_os_error());
        }

        let needed_flags = libc::IFF_RUNNING as u32;
        let blocked_flags = (libc::IFF_LOOPBACK | libc::IFF_NOARP) as u32;
        let wanted_family = libc::AF_PACKET as u16;

        let mut curr = ifaddrs;
        while !curr.is_null() {
            // Safety:
            // We know ifaddrs isn't null, otherwise we can't be in the loop. We also know we can
            // use the pointer as a reference because we got it from the kernel
            let ifaddr = unsafe { curr.as_ref().expect("curr was null when iterating NICs") };

            // Safety:
            // The pointer/data come from libc and the kernel if not null which means it can be
            // converted to a reference of the struct. This reference does not outlive the lifetime
            // of ifaddrs as it is dropped before the list is freed.
            //
            // The cast is safe to turn into a reference and dereference because the first element
            // of all sockaddr_* structures is a u16, and since all the structs are #[repr(C)] we
            // can cast a pointer from the struct to be a pointer to its first element. While this
            // is UB in C, it is perfectly fine in Rust.
            let family = match unsafe { ifaddr.ifa_addr.cast::<u16>().as_ref() } {
                Some(a) => a,
                None => continue,
            };

            if (ifaddr.ifa_flags & needed_flags) == needed_flags
                && (ifaddr.ifa_flags & blocked_flags) == 0
                && *family == wanted_family
            {
                break;
            }

            curr = ifaddr.ifa_next;
        }

        let interface = NonNull::new(curr)
            .ok_or_else(ioError::last_os_error)
            .map(|ifa| {
                // Safety:
                // The data comes from the kernel and libc which both assure that it is a valid
                // reference
                let ifaddr = unsafe { ifa.as_ref() };

                // Safety:
                // An ifaddrs's ifa_name points to a null terminated interface name so it is valid
                // to a CStr. The CStr is immediately turned into an owned string and as such
                // we don't need to worry about its lifetime past this point
                let name = unsafe { CStr::from_ptr(ifaddr.ifa_name) }
                    .to_string_lossy()
                    .to_string();

                // Safety:
                // An ifaddrs's ifa_name points to a null terminated interface name so passing it to
                // if_nametoindex is safe to do
                let index = unsafe { libc::if_nametoindex(ifaddr.ifa_name) };

                if index != 0 {
                    Some(NetworkInterface::new(name, index))
                } else {
                    None
                }
            });

        // Safety:
        // We know ifaddrs isn't null so it is safe and necessary to pass it to freeifaddrs to free
        // up the allocated memory
        unsafe { libc::freeifaddrs(ifaddrs) };

        interface
    }

    pub fn get_stats(&self) -> libc::xdp_statistics {
        let mut stats: libc::xdp_statistics;
        unsafe {
            stats = std::mem::zeroed();
            let mut optlen = size_of_val(&stats) as u32;
            libc::getsockopt(
                self.fd.as_raw_fd(),
                libc::SOL_XDP,
                libc::XDP_STATISTICS,
                &mut stats as *mut _ as _,
                &mut optlen as _,
            );
        }

        stats
    }
}

/// Attempts to get a [libc::xdp_mmap_offsets] struct from the kernel
///
/// # Returns
/// The struct if successful or an [std::io::Error] on failure
fn get_ring_offsets(fd: &XdpFd) -> Result<libc::xdp_mmap_offsets, ioError> {
    // Safety:
    // This struct is composed of simple integer offsets with invariants to maintain, its
    // content must be filled out by the kernel
    let mut offsets: libc::xdp_mmap_offsets = unsafe { std::mem::zeroed() };

    // Safety:
    // XDP_MMAP_OFFSETS is a valid operation and offsets is valid type and value to pass to it
    // for initialization
    unsafe { getsockopt(fd, libc::XDP_MMAP_OFFSETS, &mut offsets) }?;

    Ok(offsets)
}

/// A thin wrapper around [libc::setsockopt] that maps the return value to a [Result]
///
/// # Parameters:
///
/// * `op_name`: The name value of the operation to perform, e.g. [libc::XDP_UMEM_REG].
/// * `value` : Reference to the value that is to be passed.
///
/// # Safety
/// The caller must ensure that:
/// * `op_name` is a valid value to pass as the name parameter to [libc::setsockopt]
/// * `value` is a valid value to give to the operation associated with `op_name`
unsafe fn setsockopt<T>(fd: &XdpFd, op_name: i32, value: &T) -> Result<(), ioError> {
    let result = unsafe {
        libc::setsockopt(
            fd.as_raw_fd(),
            libc::SOL_XDP,
            op_name,
            value as *const _ as _,
            size_of::<T>() as _,
        )
    };

    if result == 0 {
        Ok(())
    } else {
        Err(ioError::last_os_error())
    }
}

/// A thin wrapper around [libc::getsockopt] that maps the return value to a [Result]
///
/// # Parameters:
///
/// * `op_name`: The name value of the operation to perform, e.g. [libc::XDP_UMEM_REG].
/// * `value` : Reference to a type to be written to.
///
/// # Safety
///
/// When you call this function you have to ensure that `op_name` is a valid operation value and
/// that `value` is a valid type and value to pass as the operation's output parameter.
unsafe fn getsockopt<T>(fd: &XdpFd, op_name: i32, value: &mut T) -> Result<(), ioError> {
    let result = unsafe {
        libc::getsockopt(
            fd.as_raw_fd(),
            libc::SOL_XDP,
            op_name,
            value as *mut _ as _,
            &mut (size_of::<T>() as libc::socklen_t) as *mut _,
        )
    };

    if result == 0 {
        Ok(())
    } else {
        Err(ioError::last_os_error())
    }
}

fn main() {
    env_logger::init();

    let mut sock = match XdpSock::new() {
        Ok(s) => s,
        Err(e) => {
            error!("Failed to create socket: {e}");
            return;
        }
    };

    info!("XdpSock created");
    let mut bpf = aya::Ebpf::load(aya::include_bytes_aligned!(concat!(
        env!("OUT_DIR"),
        "/ebpf_example_prog"
    )))
    .expect("Failed to construct ebpf instance");

    let program: &mut Xdp = bpf
        .program_mut("pass_all")
        .expect("Failed to find program")
        .try_into()
        .expect("Failed to convert program to Xdp program");
    program.load().expect("Failed to load program");
    program
        .attach(&sock.nic.name, XdpFlags::default())
        .expect("Failed to attach program to NIC");

    let mut map = XskMap::try_from(bpf.map_mut("XSK_MAP").expect("Failed to load map"))
            .expect("Failed to turn map into hashmap");

    map.set(0, sock.fd.as_raw_fd(), 0) // key, value, flags
        .expect("Failed to insert queue -> xsk mapping");

    // getting the 0th entry shouldn't fail
    *sock.umem.fill.data.get_mut(0).unwrap() = 0;
    sock.umem.fill.cached_prod += 1;
    sock.umem.fill.producer.store(sock.umem.fill.cached_prod, Ordering::Release);

    loop {
        let new_prod = sock.rx.producer.load(Ordering::Acquire);
        if new_prod > sock.rx.cached_prod {
            sock.rx.cached_prod = new_prod;
            // we can now read the contents of the 0th entry in the rx buffer to get info about the
            // packet in the 0th chunk of the umem buffer
            sock.rx.consumer.store(new_prod, Ordering::Release);
            // but not any more! we give up ownership once we release
            break;
        }
    }

    let data: &[u8; _] = b"\xb2X\xad&W\x16\xc4b7\x03\x01d\x08\x00E\x00\x00(\x00\x01\x00\x00@\x11\
    \xef\xfe\xc0\xa8\x04\xb9\xc0\xa8\x04\xbc\x9c\xc1c\xdd\x00\x14\xe2qhello world!";

    let chunk_start = 0;
    // not *actual* end, but end of data we're using
    let chunk_end = chunk_start + data.len();

    sock.umem.buffer[chunk_start..chunk_end].copy_from_slice(data);
    let tx_entry = sock
        .tx
        .data
        .get_mut(0)
        .expect("Getting the 0th index shouldn't fail");

    tx_entry.addr = chunk_start as _;
    tx_entry.len = data.len() as _;
    tx_entry.options = 0;

    sock.tx.cached_prod += 1;
    sock.tx
        .producer
        .store(sock.tx.cached_prod, Ordering::Release);

    unsafe {
        let ret = libc::sendto(
            sock.fd.as_raw_fd(),
            std::ptr::null(),
            0,
            libc::MSG_DONTWAIT,
            std::ptr::null(),
            0,
        );

        if ret < 0 {
            println!("sendto error: {}", std::io::Error::last_os_error());
        }
    }

    loop {
        let new_prod = sock.umem.comp.producer.load(Ordering::Acquire);
        if new_prod > sock.umem.comp.cached_prod {
            sock.umem.comp.cached_prod = new_prod;
            sock.umem
                .comp
                .consumer
                .store(new_prod, Ordering::Release);
            break;
        }
    }

    info!("stats after send: {:?}", sock.get_stats());
}
