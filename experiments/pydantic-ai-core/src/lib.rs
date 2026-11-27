mod openai_chat;
mod profile;
mod run;
mod schema;

pub use profile::model_profile;
pub use run::run_step;
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
        Some("run_step") => run_step(request["state"].as_str(), &request["event"]),
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

/// The C ABI: the same three functions serve WASM and a native shared library.
mod abi {
    #[unsafe(no_mangle)]
    pub extern "C" fn pydantic_ai_core_alloc(len: usize) -> *mut u8 {
        let mut buffer = Vec::<u8>::with_capacity(len);
        let ptr = buffer.as_mut_ptr();
        std::mem::forget(buffer);
        ptr
    }

    /// # Safety
    /// `ptr` and `len` must come from `pydantic_ai_core_alloc` or from `pydantic_ai_core_call`.
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn pydantic_ai_core_dealloc(ptr: *mut u8, len: usize) {
        drop(unsafe { Vec::from_raw_parts(ptr, 0, len) });
    }

    /// Returns the response and writes its length to `response_len`; the caller frees it with
    /// `pydantic_ai_core_dealloc`.
    ///
    /// # Safety
    /// `request` must point at `request_len` readable bytes, and `response_len` at a writable `usize`.
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn pydantic_ai_core_call(
        request: *const u8,
        request_len: usize,
        response_len: *mut usize,
    ) -> *mut u8 {
        let request =
            String::from_utf8_lossy(unsafe { std::slice::from_raw_parts(request, request_len) });
        let response = super::call(&request).into_bytes().into_boxed_slice();
        unsafe { response_len.write_unaligned(response.len()) };
        Box::into_raw(response).cast()
    }
}
