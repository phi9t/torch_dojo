use std::ffi::c_void;
struct State { calls: u64 }
unsafe extern "C" { fn dojo_dependency_add(value: f32, offset: f32) -> f32; }
#[unsafe(no_mangle)]
pub extern "C" fn fixture_new() -> *mut c_void {
    Box::into_raw(Box::new(State { calls: 0 })).cast()
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn fixture_delete(handle: *mut c_void) {
    unsafe { drop(Box::from_raw(handle.cast::<State>())); }
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn fixture_calls(handle: *mut c_void) -> u64 {
    unsafe { (*handle.cast::<State>()).calls }
}
#[unsafe(no_mangle)]
pub extern "C" fn fixture_version() -> i32 { if cfg!(version2) { 2 } else { 1 } }
#[unsafe(no_mangle)]
pub unsafe extern "C" fn fixture_apply(handle: *mut c_void, values: *mut f32, size: usize) {
    unsafe {
        (*handle.cast::<State>()).calls += 1;
        for value in std::slice::from_raw_parts_mut(values, size) {
            *value = dojo_dependency_add(*value, fixture_version() as f32);
        }
    }
}
