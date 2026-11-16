//! Port of `pydantic_ai._json_schema.JsonSchemaTransformer` and the transformers `openai_model_profile` selects.
//!
//! The control flow follows the Python closely enough that every fixture matches byte for byte.
//! The `shortcut:` comments mark known inputs where it would not; the README lists what is not covered.

use std::collections::HashMap;

use serde_json::{Map, Value, json};

type Object = Map<String, Value>;

const MAX_INLINED_SCHEMA_VALUES: usize = 250_000;
const UNION_KINDS: [&str; 3] = ["allOf", "anyOf", "oneOf"];
const DEFS_PREFIX: &str = "#/$defs/";

const STRICT_INCOMPATIBLE_KEYS: [&str; 12] = [
    "minLength",
    "maxLength",
    "patternProperties",
    "unevaluatedProperties",
    "propertyNames",
    "minProperties",
    "maxProperties",
    "unevaluatedItems",
    "contains",
    "minContains",
    "maxContains",
    "uniqueItems",
];
const STRICT_COMPATIBLE_STRING_FORMATS: [&str; 9] = [
    "date-time",
    "time",
    "date",
    "duration",
    "email",
    "hostname",
    "ipv4",
    "ipv6",
    "uuid",
];
const REGEX_LOOKAROUND_TOKENS: [&str; 4] = ["(?=", "(?!", "(?<=", "(?<!"];
const TYPE_BEARING_KEYS: [&str; 7] = ["type", "$ref", "anyOf", "oneOf", "allOf", "enum", "const"];

pub struct Transformed {
    pub schema: Value,
    pub is_strict_compatible: bool,
}

pub fn transform_schema(
    transformer: &str,
    schema: &Value,
    strict: Option<bool>,
) -> Result<Transformed, String> {
    let Value::Object(schema) = schema else {
        return Err(invalid_schema(
            "",
            "a schema must be an object or a boolean",
            schema,
        ));
    };
    let (kind, schema) = match transformer {
        "inline_defs" => (Kind::InlineDefs, schema.clone()),
        "openai" => (Kind::OpenAI, schema.clone()),
        "gpt_oss" => {
            // The outer OpenAI transformer only validates the root and resolves a boolean root definition.
            let outer = Walker::new(Kind::OpenAI, schema.clone(), strict)?;
            let mut inliner = Walker::new(Kind::InlineDefs, outer.schema, None)?;
            let inlined = Value::Object(inliner.walk()?);
            // A `$ref` the inliner did not walk to would point at a definition it dropped, so keep the schema as it was.
            let dangling = refs(&inlined)
                .iter()
                .any(|r| resolve_json_pointer(&inlined, r).is_none());
            let Value::Object(inlined) = inlined else {
                unreachable!()
            };
            (
                Kind::OpenAI,
                if dangling { inliner.schema } else { inlined },
            )
        }
        _ => return Err(format!("unknown transformer {transformer:?}")),
    };
    let mut walker = Walker::new(kind, schema, strict)?;
    let schema = Value::Object(walker.walk()?);
    Ok(Transformed {
        schema,
        is_strict_compatible: walker.is_strict_compatible,
    })
}

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    InlineDefs,
    OpenAI,
}

struct Walker {
    kind: Kind,
    schema: Object,
    strict: Option<bool>,
    is_strict_compatible: bool,
    defs: Object,
    refs_stack: Vec<String>,
    recursive_refs: Vec<String>,
    walked_defs: HashMap<String, (Object, usize)>,
    inlined_values: usize,
    root_ref: Option<String>,
}

impl Walker {
    fn new(kind: Kind, schema: Object, strict: Option<bool>) -> Result<Self, String> {
        check_schema_object(&schema, "")?;
        let defs = match schema.get("$defs") {
            Some(Value::Object(defs)) => defs.clone(),
            _ => Object::new(),
        };
        let mut walker = Self {
            kind,
            schema,
            strict,
            is_strict_compatible: true,
            defs,
            refs_stack: Vec::new(),
            recursive_refs: Vec::new(),
            walked_defs: HashMap::new(),
            inlined_values: 0,
            root_ref: None,
        };
        if kind == Kind::OpenAI
            && let Some(root_ref) = walker.schema.get("$ref").and_then(Value::as_str)
        {
            match walker.defs.get(&definition_key(root_ref)) {
                Some(Value::Bool(false)) => {
                    return Err(
                        "OpenAI does not support boolean `false` root schemas, which reject all inputs.".to_owned()
                    );
                }
                // Resolve a boolean root before the walk can rewrite references to it to `#`.
                Some(Value::Bool(true)) => {
                    walker.schema.shift_remove("$ref");
                }
                _ => walker.root_ref = Some(root_ref.to_owned()),
            }
        }
        Ok(walker)
    }

    fn inlines_defs(&self) -> bool {
        self.kind == Kind::InlineDefs
    }

    fn walk(&mut self) -> Result<Object, String> {
        let mut result = self.walk_base()?;
        if self.kind == Kind::OpenAI {
            // Draft-07 `definitions` are referenced like `$defs`, so their entries need the same handling.
            if let Some(Value::Object(definitions)) = result.get_mut("definitions") {
                for definition in definitions.values_mut() {
                    *definition = self.handle(definition.take())?;
                }
            }
            if let Some(root_ref) = &self.root_ref {
                // References to the root were rewritten to `#`, so the root definition is merged into the root.
                result.shift_remove("$ref");
                if let Some(Value::Object(root)) = self.defs.get(&definition_key(root_ref)) {
                    result.extend(root.clone());
                }
            }
        }
        Ok(result)
    }

    fn walk_base(&mut self) -> Result<Object, String> {
        let mut schema = self.schema.clone();
        self.recursive_refs.clear();
        self.walked_defs.clear();
        self.inlined_values = 0;

        schema.shift_remove("$defs");
        let Value::Object(mut handled) = self.handle(Value::Object(schema))? else {
            unreachable!()
        };

        if !self.inlines_defs() && !self.defs.is_empty() {
            let mut handled_defs = Object::new();
            for (key, definition) in self.defs.clone() {
                let handled_def = self.handle(definition.clone())?;
                self.defs
                    .insert(key.clone(), after_in_place_walk(&definition, &handled_def));
                handled_defs.insert(key, handled_def);
            }
            handled.insert("$defs".to_owned(), Value::Object(handled_defs));
        } else if !self.recursive_refs.is_empty() {
            // Recursive refs can't be inlined, so they need a `$defs` + `$ref` structure.
            let mut defs = Object::new();
            for key in self.recursive_refs.clone() {
                let walked = self.copy_walked_def(&key)?;
                defs.insert(key, Value::Object(walked));
            }
            let root_ref = self
                .schema
                .get("$ref")
                .and_then(Value::as_str)
                .map(str::to_owned);
            let root_key = match &root_ref {
                Some(root_ref) => definition_key(root_ref),
                None => {
                    let mut root_key = self
                        .schema
                        .get("title")
                        .and_then(Value::as_str)
                        .unwrap_or("root")
                        .to_owned();
                    while defs.contains_key(&root_key) {
                        root_key.push_str("_root");
                    }
                    root_key
                }
            };
            defs.insert(root_key.clone(), Value::Object(handled));
            let root_ref = root_ref
                .filter(|r| !r.is_empty())
                .unwrap_or_else(|| format!("{DEFS_PREFIX}{root_key}"));
            return Ok(Object::from_iter([
                ("$defs".to_owned(), Value::Object(defs)),
                ("$ref".to_owned(), json!(root_ref)),
            ]));
        }
        Ok(handled)
    }

    fn handle(&mut self, node: Value) -> Result<Value, String> {
        let Value::Object(mut schema) = node else {
            return Ok(node);
        };

        if self.inlines_defs()
            && let Some(reference) = schema
                .get("$ref")
                .and_then(Value::as_str)
                .filter(|r| !r.is_empty())
            && let Some(inlined) = self.inlined_ref(&schema, reference)?
        {
            return Ok(Value::Object(inlined));
        }

        // Object and array keywords are only walked when `type` is exactly `'object'` or `'array'`.
        let schema_type = schema.get("type").cloned().unwrap_or(Value::Null);
        if schema_type == "object" {
            self.handle_object(&mut schema)?;
        } else if schema_type == "array" {
            self.handle_array(&mut schema)?;
        } else if schema_type.is_null() {
            for union_kind in UNION_KINDS {
                schema = self.handle_union(schema, union_kind)?;
            }
        }
        if !schema_type.is_null() {
            for union_kind in UNION_KINDS {
                if let Some(Value::Array(members)) = schema.get_mut(union_kind) {
                    self.handle_each(members.iter_mut())?;
                }
            }
        }
        self.transform(&mut schema)?;
        Ok(Value::Object(schema))
    }

    fn handle_each<'a>(
        &mut self,
        nodes: impl Iterator<Item = &'a mut Value>,
    ) -> Result<(), String> {
        for node in nodes {
            *node = self.handle(node.take())?;
        }
        Ok(())
    }

    /// The walked definition `reference` points at, with the other keywords of `schema` merged over it.
    ///
    /// `None` when the `$ref` stays, as a recursive one does.
    fn inlined_ref(&mut self, schema: &Object, reference: &str) -> Result<Option<Object>, String> {
        let key = match reference.strip_prefix(DEFS_PREFIX) {
            Some(name) => unescape_pointer_token(name),
            None => reference.to_owned(),
        };
        if self.refs_stack.contains(&key) {
            if !self.defs.contains_key(&key) {
                return Err(format!(
                    "Recursive JSON pointer `$ref` {} is not supported when inlining definitions",
                    py_repr(&json!(key))
                ));
            }
            if !self.recursive_refs.contains(&key) {
                self.recursive_refs.push(key);
            }
            return Ok(None);
        }
        if self.recursive_refs.contains(&key) {
            return Ok(None);
        }
        // Keywords alongside the `$ref`, like a field-level `description`, must survive inlining.
        let siblings: Object = schema
            .iter()
            .filter(|(k, _)| *k != "$ref")
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        if siblings.is_empty() {
            self.copy_walked_def(&key).map(Some)
        } else {
            self.walk_def(&key, siblings).map(Some)
        }
    }

    /// A copy of the definition `key` refers to, walked once and cached with the number of values it counted.
    fn copy_walked_def(&mut self, key: &str) -> Result<Object, String> {
        if let Some((walked, values)) = self.walked_defs.get(key) {
            let (walked, values) = (walked.clone(), *values);
            self.count_inlined(values)?;
            return Ok(walked);
        }
        let start = self.inlined_values;
        let walked = self.walk_def(key, Object::new())?;
        self.walked_defs.insert(
            key.to_owned(),
            (walked.clone(), self.inlined_values - start),
        );
        Ok(walked)
    }

    fn walk_def(&mut self, key: &str, siblings: Object) -> Result<Object, String> {
        let definition = match self
            .defs
            .get(key)
            .filter(|definition| !definition.is_null())
        {
            Some(definition) => definition.clone(),
            None => {
                let root = Value::Object(self.schema.clone());
                let Some(definition) = resolve_json_pointer(&root, key) else {
                    return Err(format!("Could not find $ref definition for {key}"));
                };
                // A pointer can reach nodes the up-front check doesn't visit, like draft-07 `definitions`.
                check_schema_object(definition, &json_pointer_path(key))?;
                Value::Object(definition.clone())
            }
        };

        self.refs_stack.push(key.to_owned());
        let schema = match definition {
            Value::Object(mut definition) => {
                definition.extend(siblings);
                definition
            }
            // A false definition must reject even when a sibling supplies its own `not`.
            Value::Bool(false) => {
                let mut schema = siblings;
                schema.insert("not".to_owned(), json!({}));
                schema
            }
            _ => siblings,
        };
        self.count_inlined(count_values(&schema))?;
        let Value::Object(walked) = self.handle(Value::Object(schema))? else {
            unreachable!()
        };
        self.refs_stack.pop();
        Ok(walked)
    }

    fn count_inlined(&mut self, values: usize) -> Result<(), String> {
        self.inlined_values += values;
        if self.inlined_values > MAX_INLINED_SCHEMA_VALUES {
            return Err(
                "JSON schema is too large: its `$ref`s expand to more than 250,000 JSON values. \
                 Refer to large definitions from fewer places."
                    .to_owned(),
            );
        }
        Ok(())
    }

    fn handle_object(&mut self, schema: &mut Object) -> Result<(), String> {
        if let Some(Value::Object(properties)) = schema.get_mut("properties") {
            self.handle_each(properties.values_mut())?;
        }
        if let Some(additional_properties @ Value::Object(_)) =
            schema.get_mut("additionalProperties")
        {
            *additional_properties = self.handle(additional_properties.take())?;
        }
        if let Some(Value::Object(pattern_properties)) = schema.get_mut("patternProperties") {
            self.handle_each(pattern_properties.values_mut())?;
        }
        self.handle_keyword_subschema(schema, "propertyNames")
    }

    fn handle_array(&mut self, schema: &mut Object) -> Result<(), String> {
        if let Some(Value::Array(prefix_items)) = schema.get_mut("prefixItems") {
            self.handle_each(prefix_items.iter_mut())?;
        }
        match schema.get_mut("items") {
            // Drafts before 2020-12 spell a tuple as an `items` list.
            Some(Value::Array(items)) => self.handle_each(items.iter_mut())?,
            Some(items @ Value::Object(_)) if is_truthy(items) => {
                *items = self.handle(items.take())?
            }
            _ => {}
        }
        self.handle_keyword_subschema(schema, "contains")
    }

    /// Walk a `propertyNames` or `contains` subschema, which only an inlining transformer does.
    fn handle_keyword_subschema(&mut self, schema: &mut Object, key: &str) -> Result<(), String> {
        if self.inlines_defs()
            && let Some(subschema) = schema.get_mut(key).filter(|subschema| !subschema.is_null())
        {
            *subschema = self.handle(subschema.take())?;
        }
        Ok(())
    }

    fn handle_union(&mut self, mut schema: Object, union_kind: &str) -> Result<Object, String> {
        let Some(members) = schema.shift_remove(union_kind) else {
            return Ok(schema);
        };
        let Value::Array(mut handled) = members else {
            return Err(invalid_schema(
                union_kind,
                &format!("`{union_kind}` must be a list of schemas"),
                &members,
            ));
        };
        self.handle_each(handled.iter_mut())?;

        // A single member replaces the union, unless it rejects: a sibling `not` would overwrite its constraint.
        if let [Value::Object(only)] = handled.as_mut_slice()
            && only.get("not") != Some(&json!({}))
        {
            let mut merged = std::mem::take(only);
            merged.extend(schema);
            return Ok(merged);
        }
        schema.insert(union_kind.to_owned(), Value::Array(handled));
        Ok(schema)
    }

    fn transform(&mut self, schema: &mut Object) -> Result<(), String> {
        if self.kind == Kind::InlineDefs {
            return Ok(());
        }
        for key in ["title", "$schema", "discriminator"] {
            schema.shift_remove(key);
        }

        // `default` is not allowed in strict mode, but helps some models when not strict.
        if schema.contains_key("default") {
            match self.strict {
                Some(true) => drop(schema.shift_remove("default")),
                None => self.is_strict_compatible = false,
                Some(false) => {}
            }
        }

        if let Some(reference) = schema
            .get("$ref")
            .and_then(Value::as_str)
            .filter(|r| !r.is_empty())
            .map(str::to_owned)
        {
            if Some(reference.as_str()) == self.root_ref.as_deref() {
                schema.insert("$ref".to_owned(), json!("#"));
            } else if self.strict.is_none()
                && reference.starts_with("#/")
                && !is_top_level_definition_ref(&reference)
            {
                // A pointer elsewhere in the schema, like `#/properties/from`, is rejected in strict mode.
                self.is_strict_compatible = false;
            }
            if schema.len() > 1 {
                // Strict mode doesn't support siblings to `$ref`, but does allow siblings to `anyOf`.
                let reference = schema.shift_remove("$ref");
                schema.insert("anyOf".to_owned(), json!([{"$ref": reference}]));
            }
        }

        self.transform_strict_incompatible_values(schema);

        if schema.contains_key("oneOf") {
            if self.strict == Some(true) {
                let members = schema.shift_remove("oneOf").unwrap_or_default();
                schema.insert("anyOf".to_owned(), members);
            } else {
                self.is_strict_compatible = false;
            }
        }

        match schema.get("type").and_then(Value::as_str) {
            Some("object") => self.transform_object(schema),
            Some("array") => self.check_array_items(schema)?,
            _ => {}
        }
        Ok(())
    }

    fn transform_strict_incompatible_values(&mut self, schema: &mut Object) {
        let mut incompatible: Vec<(&str, Value)> = Vec::new();
        for key in STRICT_INCOMPATIBLE_KEYS {
            if let Some(value) = schema.get(key) {
                incompatible.push((key, value.clone()));
            }
        }
        if let Some(format) = schema.get("format").filter(|format| is_truthy(format))
            && !STRICT_COMPATIBLE_STRING_FORMATS
                .iter()
                .any(|known| format == known)
        {
            incompatible.push(("format", format.clone()));
        }
        if let Some(Value::String(pattern)) = schema.get("pattern")
            && regex_contains_lookaround(pattern)
        {
            incompatible.push(("pattern", json!(pattern)));
        }
        if incompatible.is_empty() {
            return;
        }
        match self.strict {
            Some(true) => {
                let description = schema
                    .get("description")
                    .filter(|description| is_truthy(description))
                    .map(py_str);
                let mut notes = Vec::new();
                for (key, value) in incompatible {
                    schema.shift_remove(key);
                    notes.push(format!("{key}={}", py_str(&value)));
                }
                let notes = notes.join(", ");
                let description = match description {
                    Some(description) => format!("{description} ({notes})"),
                    None => notes,
                };
                schema.insert("description".to_owned(), json!(description));
            }
            None => self.is_strict_compatible = false,
            Some(false) => {}
        }
    }

    fn transform_object(&mut self, schema: &mut Object) {
        // OpenAI drops objects without `properties`.
        if !schema.contains_key("properties") {
            schema.insert("properties".to_owned(), json!({}));
        }
        let property_names: Vec<Value> = match schema.get("properties") {
            Some(Value::Object(properties)) => properties.keys().map(|key| json!(key)).collect(),
            _ => Vec::new(),
        };
        match self.strict {
            Some(true) => {
                schema.insert("additionalProperties".to_owned(), json!(false));
                schema.insert("required".to_owned(), Value::Array(property_names));
            }
            None => {
                match schema.get("additionalProperties") {
                    None | Some(Value::Null | Value::Bool(false)) => {
                        schema.insert("additionalProperties".to_owned(), json!(false));
                    }
                    Some(_) => self.is_strict_compatible = false,
                }
                match schema.get("required") {
                    Some(Value::Array(required))
                        if property_names.iter().all(|name| required.contains(name)) => {}
                    // Python only reads `required` when there is a property to look up in it.
                    Some(_) if property_names.is_empty() => {}
                    // shortcut: a string or object `required` on a node with properties counts as incompatible;
                    // Python does a substring or key lookup, reachable only under draft-07 `definitions`.
                    _ => self.is_strict_compatible = false,
                }
            }
            Some(false) => {}
        }
    }

    /// Strict mode needs `items` to type the elements: a boolean node, `{}` or a metadata-only node does not.
    fn check_array_items(&mut self, schema: &Object) -> Result<(), String> {
        let has_typed_items = matches!(
            schema.get("items"),
            Some(Value::Object(items)) if TYPE_BEARING_KEYS.iter().any(|key| items.contains_key(*key))
        );
        if has_typed_items || schema.get("prefixItems").is_some_and(is_truthy) {
            return Ok(());
        }
        match self.strict {
            Some(true) => Err("OpenAI strict mode requires array items to have a type, but got an untyped array \
                 (e.g. a bare `list`). Add a type parameter such as `list[str]`, or set `strict=False`."
                .to_owned()),
            None => {
                self.is_strict_compatible = false;
                Ok(())
            }
            Some(false) => Ok(()),
        }
    }
}

/// What Python leaves in `self.defs` after walking a definition.
///
/// The walk edits a node in place, except a typeless node with a union: that one is rebuilt, and the original only
/// loses the union key. `OpenAIJsonSchemaTransformer.walk()` reads the original back for the root definition.
fn after_in_place_walk(original: &Value, handled: &Value) -> Value {
    if let Value::Object(original) = original
        && original.get("type").is_none_or(Value::is_null)
        && let Some(union_kind) = UNION_KINDS
            .iter()
            .find(|kind| original.contains_key(**kind))
    {
        let mut original = original.clone();
        original.shift_remove(*union_kind);
        return Value::Object(original);
    }
    handled.clone()
}

fn check_schema_shape(node: &Value, path: &str) -> Result<(), String> {
    match node {
        Value::Bool(_) => Ok(()),
        Value::Object(schema) => check_schema_object(schema, path),
        _ => Err(invalid_schema(
            path,
            "a schema must be an object or a boolean",
            node,
        )),
    }
}

/// Fail naming the offending node if `schema` is not shaped like the JSON Schema the transformers read.
fn check_schema_object(schema: &Object, path: &str) -> Result<(), String> {
    let child = |key: &str| {
        if path.is_empty() {
            key.to_owned()
        } else {
            format!("{path}.{key}")
        }
    };
    let present = |key: &str| schema.get(key).filter(|value| !value.is_null());

    for key in ["additionalProperties", "items", "propertyNames", "contains"] {
        match present(key) {
            Some(Value::Array(members)) if key == "items" => {
                for (index, member) in members.iter().enumerate() {
                    check_schema_shape(member, &format!("{}.{index}", child(key)))?;
                }
            }
            Some(value) => check_schema_shape(value, &child(key))?,
            None => {}
        }
    }
    for (key, is_object) in [
        ("properties", true),
        ("patternProperties", true),
        ("$defs", true),
        ("prefixItems", false),
        ("allOf", false),
        ("anyOf", false),
        ("oneOf", false),
    ] {
        match (present(key), is_object) {
            (None, _) => {}
            (Some(Value::Object(members)), true) => {
                for (name, member) in members {
                    check_schema_shape(member, &format!("{}.{name}", child(key)))?;
                }
            }
            (Some(Value::Array(members)), false) => {
                for (index, member) in members.iter().enumerate() {
                    check_schema_shape(member, &format!("{}.{index}", child(key)))?;
                }
            }
            (Some(value), _) => {
                let kind = if is_object { "an object" } else { "a list" };
                return Err(invalid_schema(
                    &child(key),
                    &format!("`{key}` must be {kind} of schemas"),
                    value,
                ));
            }
        }
    }
    let is_type = |value: &Value| match value {
        Value::String(_) => true,
        Value::Array(types) => types.iter().all(Value::is_string),
        _ => false,
    };
    let is_integer = |value: &Value| {
        value.is_i64() || value.is_u64() || value.as_f64().is_some_and(|n| n.fract() == 0.0)
    };
    let checks: [(&str, &dyn Fn(&Value) -> bool, &str); 5] = [
        ("$ref", &Value::is_string, "must be a string"),
        ("type", &is_type, "must be a string or a list of strings"),
        ("required", &Value::is_array, "must be a list"),
        ("enum", &Value::is_array, "must be a list"),
        ("minItems", &is_integer, "must be an integer"),
    ];
    for (key, is_well_formed, what) in checks {
        if let Some(value) = present(key).filter(|value| !is_well_formed(value)) {
            return Err(invalid_schema(
                &child(key),
                &format!("`{key}` {what}"),
                value,
            ));
        }
    }
    Ok(())
}

fn invalid_schema(path: &str, what: &str, value: &Value) -> String {
    let path = if path.is_empty() { "<root>" } else { path };
    format!(
        "Invalid JSON Schema at `{path}`: {what}, got {}",
        py_repr(value)
    )
}

/// The name of the definition a `$ref` like `#/$defs/a~1b` points at.
fn definition_key(reference: &str) -> String {
    unescape_pointer_token(reference.strip_prefix(DEFS_PREFIX).unwrap_or(reference))
}

fn unescape_pointer_token(token: &str) -> String {
    token.replace("~1", "/").replace("~0", "~")
}

fn json_pointer_path(reference: &str) -> String {
    reference
        .split('/')
        .skip(1)
        .map(unescape_pointer_token)
        .collect::<Vec<_>>()
        .join(".")
}

/// The schema object a local JSON-pointer `$ref` (RFC 6901) like `#/properties/from` points at, if any.
fn resolve_json_pointer<'a>(schema: &'a Value, reference: &str) -> Option<&'a Object> {
    if reference != "#" && !reference.starts_with("#/") {
        return None;
    }
    let mut node = schema;
    for token in reference.split('/').skip(1) {
        let token = unescape_pointer_token(token);
        node = match node {
            Value::Object(object) => object.get(&token)?,
            // An array index is written in canonical decimal: ASCII digits, no leading zero.
            Value::Array(array) => array.get(
                token
                    .parse::<usize>()
                    .ok()
                    .filter(|index| index.to_string() == token)?,
            )?,
            _ => return None,
        };
    }
    node.as_object()
}

/// Every `$ref` in `node`, at any depth.
fn refs(node: &Value) -> Vec<&str> {
    match node {
        Value::Object(object) => object
            .iter()
            .flat_map(|(key, value)| match value {
                Value::String(reference) if key == "$ref" => vec![reference.as_str()],
                _ => refs(value),
            })
            .collect(),
        Value::Array(array) => array.iter().flat_map(refs).collect(),
        _ => Vec::new(),
    }
}

fn count_values(schema: &Object) -> usize {
    fn count(value: &Value) -> usize {
        1 + match value {
            Value::Object(object) => object.values().map(count).sum(),
            Value::Array(array) => array.iter().map(count).sum(),
            _ => 0,
        }
    }
    1 + schema.values().map(count).sum::<usize>()
}

fn is_top_level_definition_ref(reference: &str) -> bool {
    [DEFS_PREFIX, "#/definitions/"]
        .iter()
        .filter_map(|prefix| reference.strip_prefix(prefix))
        .any(|name| !name.is_empty() && !name.contains('/'))
}

fn regex_contains_lookaround(pattern: &str) -> bool {
    let mut escaped = false;
    for (index, char) in pattern.char_indices() {
        if escaped {
            escaped = false;
        } else if char == '\\' {
            escaped = true;
        } else if REGEX_LOOKAROUND_TOKENS
            .iter()
            .any(|token| pattern[index..].starts_with(token))
        {
            return true;
        }
    }
    false
}

/// Python truthiness of a JSON value.
fn is_truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(value) => *value,
        Value::Number(number) => number.as_f64() != Some(0.0),
        Value::String(string) => !string.is_empty(),
        Value::Array(array) => !array.is_empty(),
        Value::Object(object) => !object.is_empty(),
    }
}

/// Python `str()` of a JSON value: strict mode moves dropped keywords into `description` formatted that way.
fn py_str(value: &Value) -> String {
    match value {
        Value::String(string) => string.clone(),
        _ => py_repr(value),
    }
}

/// Python `repr()` of a JSON value.
fn py_repr(value: &Value) -> String {
    match value {
        Value::Null => "None".to_owned(),
        Value::Bool(true) => "True".to_owned(),
        Value::Bool(false) => "False".to_owned(),
        // shortcut: floats print as serde prints them, which differs from Python for exponents like `1e-07`.
        Value::Number(number) => number.to_string(),
        Value::String(string) => py_repr_str(string),
        Value::Array(array) => format!(
            "[{}]",
            array.iter().map(py_repr).collect::<Vec<_>>().join(", ")
        ),
        Value::Object(object) => {
            let items: Vec<String> = object
                .iter()
                .map(|(k, v)| format!("{}: {}", py_repr_str(k), py_repr(v)))
                .collect();
            format!("{{{}}}", items.join(", "))
        }
    }
}

fn py_repr_str(string: &str) -> String {
    let quote = if string.contains('\'') && !string.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut repr = String::from(quote);
    for char in string.chars() {
        match char {
            '\\' => repr.push_str("\\\\"),
            '\n' => repr.push_str("\\n"),
            '\r' => repr.push_str("\\r"),
            '\t' => repr.push_str("\\t"),
            _ if char == quote => {
                repr.push('\\');
                repr.push(char);
            }
            // shortcut: only ASCII control characters are escaped; Python also escapes non-printable Unicode.
            _ if char.is_ascii_control() => repr.push_str(&format!("\\x{:02x}", char as u32)),
            _ => repr.push(char),
        }
    }
    repr.push(quote);
    repr
}
