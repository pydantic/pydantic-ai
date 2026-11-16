mod profile;
mod schema;

pub use profile::model_profile;
pub use schema::{Transformed, transform_schema};

use serde_json::{Value, json};

/// JSON in, JSON out: the one entry point every language binding goes through.
pub fn call(request: &str) -> String {
    let result = serde_json::from_str::<Value>(request)
        .map_err(|e| e.to_string())
        .and_then(|r| dispatch(&r));
    match result {
        Ok(value) => json!({"ok": value}),
        Err(error) => json!({"error": error}),
    }
    .to_string()
}

fn dispatch(request: &Value) -> Result<Value, String> {
    let text = |name: &str| {
        request[name]
            .as_str()
            .ok_or_else(|| format!("`{name}` must be a string"))
    };
    match request["fn"].as_str() {
        Some("model_profile") => model_profile(text("family")?, text("model")?),
        Some("transform_schema") => {
            let strict = match &request["strict"] {
                Value::Null => None,
                Value::Bool(strict) => Some(*strict),
                _ => return Err("`strict` must be a boolean or null".to_owned()),
            };
            let out = transform_schema(text("transformer")?, &request["schema"], strict)?;
            Ok(json!({"schema": out.schema, "is_strict_compatible": out.is_strict_compatible}))
        }
        other => Err(format!("unknown fn {other:?}")),
    }
}

#[cfg(target_arch = "wasm32")]
mod wasm {
    #[unsafe(no_mangle)]
    pub extern "C" fn alloc(len: usize) -> *mut u8 {
        let mut buffer = Vec::<u8>::with_capacity(len);
        let ptr = buffer.as_mut_ptr();
        std::mem::forget(buffer);
        ptr
    }

    /// # Safety
    /// `ptr` and `len` must come from `alloc` or from the result of `call`.
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn dealloc(ptr: *mut u8, len: usize) {
        drop(unsafe { Vec::from_raw_parts(ptr, 0, len) });
    }

    /// Returns the response as `(ptr << 32) | len`; the caller frees it with `dealloc`.
    ///
    /// # Safety
    /// `ptr` must point at `len` readable bytes.
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn call(ptr: *const u8, len: usize) -> u64 {
        let request = String::from_utf8_lossy(unsafe { std::slice::from_raw_parts(ptr, len) });
        let response = super::call(&request).into_bytes().into_boxed_slice();
        let packed = ((response.as_ptr() as u64) << 32) | response.len() as u64;
        std::mem::forget(response);
        packed
    }
}
