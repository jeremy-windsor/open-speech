// A stand-in for a view module with a singleton (tests/js/core/harness-fakes.test.js): importFresh() must
// call its __resetForTests().
let count = 0;
export function bump() { count += 1; return count; }
export function value() { return count; }
export function __resetForTests() { count = 0; }
