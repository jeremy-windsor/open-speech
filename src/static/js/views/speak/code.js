/**
 * Speak: copy the current request as code. Pure. None of these ever embeds the real API key:
 * with a key set they read it from the environment, and without one they send no Authorization.
 *
 *   toCurl(req, {origin, hasKey})   curl -sS -X POST "<origin>/v1/audio/speech" … -o speech.<fmt>
 *   toPython(req, {origin, hasKey}) the requests library, key from os.environ["OPEN_SPEECH_API_KEY"]
 *   toJs(req, {origin, hasKey})     fetch, key from process.env.OPEN_SPEECH_API_KEY
 *
 * `req` is buildSpeechRequest()'s result (only its body is used: a copied request always writes a file,
 * so it never asks for a stream).
 * @module views/speak/code
 */

const SHELL_KEY = '$OPEN_SPEECH_API_KEY';

function endpoint(origin) {
  return `${String(origin || '').replace(/\/+$/, '')}/v1/audio/speech`;
}

function bodyOf(req) {
  return (req && req.body) || {};
}

function extension(req) {
  const fmt = String(bodyOf(req).response_format || 'mp3').replace(/[^a-z0-9]/gi, '');
  return fmt || 'mp3';
}

/** Single-quote a string for a POSIX shell: ' becomes '\''. */
export function shellQuote(text) {
  return `'${String(text).replace(/'/g, "'\\''")}'`;
}

/** curl, with the JSON body in single quotes (single quotes inside it escaped for POSIX shells). */
export function toCurl(req, { origin = '', hasKey = false } = {}) {
  const lines = [`curl -sS -X POST "${endpoint(origin)}"`, '  -H "Content-Type: application/json"'];
  if (hasKey) lines.push(`  -H "Authorization: Bearer ${SHELL_KEY}"`);
  lines.push(`  -d ${shellQuote(JSON.stringify(bodyOf(req)))}`);
  lines.push(`  -o speech.${extension(req)}`);
  return lines.join(' \\\n');
}

/** A JSON value as a Python literal (None/True/False; JSON string escapes are valid Python). */
export function pyLiteral(value, indent = 0) {
  const pad = '    '.repeat(indent + 1);
  const end = '    '.repeat(indent);
  if (value === null || value === undefined) return 'None';
  if (value === true) return 'True';
  if (value === false) return 'False';
  if (typeof value === 'number') return Number.isFinite(value) ? String(value) : 'None';
  if (typeof value === 'string') return JSON.stringify(value);
  if (Array.isArray(value)) {
    if (!value.length) return '[]';
    return `[\n${value.map((v) => `${pad}${pyLiteral(v, indent + 1)},`).join('\n')}\n${end}]`;
  }
  const keys = Object.keys(value);
  if (!keys.length) return '{}';
  return `{\n${keys.map((k) => `${pad}${JSON.stringify(k)}: ${pyLiteral(value[k], indent + 1)},`).join('\n')}\n${end}}`;
}

/** Python with requests; the key comes from the environment. */
export function toPython(req, { origin = '', hasKey = false } = {}) {
  const lines = [];
  if (hasKey) lines.push('import os', '');
  lines.push('import requests', '');
  lines.push('response = requests.post(');
  lines.push(`    ${JSON.stringify(endpoint(origin))},`);
  if (hasKey) lines.push('    headers={"Authorization": "Bearer " + os.environ["OPEN_SPEECH_API_KEY"]},');
  lines.push(`    json=${pyLiteral(bodyOf(req), 1)},`);
  lines.push('    timeout=300,');
  lines.push(')');
  lines.push('response.raise_for_status()');
  lines.push(`with open("speech.${extension(req)}", "wb") as f:`);
  lines.push('    f.write(response.content)');
  return lines.join('\n');
}

/** JavaScript with fetch (Node 18+ or a browser on the same origin); the key comes from the environment. */
export function toJs(req, { origin = '', hasKey = false } = {}) {
  const headers = ['    "Content-Type": "application/json",'];
  if (hasKey) headers.push('    Authorization: `Bearer ${process.env.OPEN_SPEECH_API_KEY}`,');
  const body = JSON.stringify(bodyOf(req), null, 2).split('\n').map((line, i) => (i ? `  ${line}` : line)).join('\n');
  return [
    // Plain strings, not a template: the generated text names fetch, which the architecture lint reserves.
    'const response = await fetch(' + JSON.stringify(endpoint(origin)) + ', {',
    '  method: "POST",',
    '  headers: {',
    ...headers,
    '  },',
    `  body: JSON.stringify(${body}),`,
    '});',
    'if (!response.ok) throw new Error(`HTTP ${response.status}`);',
    `const audio = await response.blob(); // ${extension(req).toUpperCase()} audio`,
  ].join('\n');
}
