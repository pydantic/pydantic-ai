import assert from 'node:assert/strict'
import { test } from 'node:test'
import { compact } from '../src/format.ts'

test('compact keeps values under one million on the k scale', () => {
  assert.equal(compact(null), '?')
  assert.equal(compact(undefined), '?')
  assert.equal(compact(42), '42')
  assert.equal(compact(1234), '1.2k')
  assert.equal(compact(7700), '7.7k')
  assert.equal(compact(56789), '57k')
  assert.equal(compact(200_000), '200k')
  assert.equal(compact(999_999), '1000k')
})

test('compact uses M at one million tokens and above', () => {
  assert.equal(compact(1_000_000), '1M')
  assert.equal(compact(1_048_576), '1M')
  assert.equal(compact(1_500_000), '1.5M')
})
