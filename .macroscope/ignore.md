---
ignoreTests: false
---

# Repository-wide Macroscope ignore (code review + any check-run agents).
#
# A custom ignore file REPLACES Macroscope's built-in defaults, so this copies
# their default "base" patterns verbatim to preserve them, then adds this repo's
# own recorded/generated files on top. https://docs.macroscope.com/
#
# Test code is reviewed because `ignoreTests` is false. Recorded cassettes and
# binary fixtures under tests/ remain ignored by the patterns below.

# ---- Macroscope default base patterns (copied to preserve them) ----
**/.git/**
**/__pycache__/**
**/.pytest_cache/**
**/.mypy_cache/**
**/.ruff_cache/**
**/venv/**
**/.venv/**
**/node_modules/**
**/site-packages/**
**/.pnpm-store/**
**/__Snapshots__/**
**/__snapshots__/**
**/.agents/skills/**
**/.claude/skills/**
**/.github/skills/**
**/bower_components/**
**/jspm_packages/**
**/.next/**
**/.svelte-kit/**
**/.nuxt/**
**/.output/**
**/.vercel/**
**/.angular/**
**/vendor/**
**/_vendor/**
**/third_party/**
**/Pods/**
**/.bundle/**
build/**
out/**
env/**
ENV/**
**/target/**
**/dist/**
**/generated/**
**/intermediates/**
**/generated_sources/**
**/generated-sources/**
**/generated-src/**
**/src/main/generated/**
**/*.min.js
**/*.min.css
**/*.bundle.js
**/.pnp.cjs
**/.pnp.loader.mjs
**/*_pb.d.ts
**/*_pb.js
**/*.pb.go
**/*_pb2.py
**/*_pb2_grpc.py
**/*_pb2.pyi
**/*.grpc.swift
**/*.pb.swift
**/*.sql.go
**/*.designer.cs
**/*.g.dart
**/*.pb.dart
**/*_pb.rb
**/*.d.ts
**/*.gen.ts
**/*.gen.tsx
**/*.gen.js
**/*.gen.jsx
**/go.mod
**/package.json
**/*.pbxproj
**/*.xcstrings
**/*.strings
**/*.properties
**/pom.xml
**/Package.swift
**/bun.lock
**/.eslintrc
**/.eslintignore
**/go.sum
**/package-lock.json
**/pnpm-lock.yaml
**/yarn.lock
**/Package.resolved
**/*.jpg
**/*.jpeg
**/*.png
**/*.gif
**/*.svg
**/*.ico
**/*.webp
**/*.bmp
**/*.tiff
**/*.woff
**/*.woff2
**/*.ttf
**/*.eot
**/*.otf
**/*.mp3
**/*.mp4
**/*.wav
**/*.avi
**/*.mov
**/*.mkv
**/*.flac
**/*.ogg
**/*.srt
**/*.zip
**/*.tar
**/*.gz
**/*.rar
**/*.7z
**/*.bz2
**/*.pdf
**/*.doc
**/*.docx
**/*.xls
**/*.xlsx
**/*.ppt
**/*.pptx
**/*.db
**/*.sqlite
**/*.sqlite3
**/*.parquet
**/*.avro
**/*.arrow
**/*.npy
**/*.pkl
**/*.jsonl
**/*.onnx
**/*.tflite
**/*.h5
**/*.safetensors
**/*.exe
**/*.dll
**/*.so
**/*.dylib
**/*.bin
**/*.pyc
**/*.class
**/*.o
**/*.a
**/*.wasm
**/*.cer
**/*.pem
**/*.p12
**/*.stringsdict
**/*.snap
**/*.adoc
**/*.arb
**/*.lock
**/*.po
**/*.fbx
**/*.log
**/*.xib
**/*.meta
**/*.kml
**/*.prefab
**/*.eml
**/*.csv
**/*.grpc.reflection
**/*.js.map

# ---- This repo: recorded + generated files the defaults do not cover ----

# Recorded VCR cassettes (LLM/HTTP recordings, ~1000 files / ~200 MB). These
# live under tests/, so keep them ignored explicitly now that test code is
# reviewed.
**/cassettes/**

# Compiled gh-aw workflows (generated from their `.md` sources).
.github/workflows/*.lock.yml
