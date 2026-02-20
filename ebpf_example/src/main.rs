#![no_std]
#![no_main]

use aya_ebpf::{
    bindings::xdp_action,
    macros::{map, xdp},
    maps::XskMap,
    programs::XdpContext,
};

#[map]
static XSK_MAP: XskMap = XskMap::with_max_entries(64, 0);

#[xdp]
pub fn pass_all(ctx: XdpContext) -> u32 {
    XSK_MAP
        .redirect(get_queue_idx(&ctx), 0)
        .unwrap_or(xdp_action::XDP_PASS)
}

#[inline(always)]
fn get_queue_idx(ctx: &XdpContext) -> u32 {
    unsafe { *ctx.ctx }.rx_queue_index
}

#[cfg(not(test))]
#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    loop {}
}

#[unsafe(link_section = "license")]
#[unsafe(no_mangle)]
static LICENSE: [u8; 4] = *b"GPL\0";
