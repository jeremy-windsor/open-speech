// A small HTML parser for well-formed markup (partials, shells, templates): quoted, unquoted and boolean
// attributes, the common entities, void elements, raw text in script/style, RCDATA in textarea/title,
// inline SVG (case kept, '/>' closes), comments ignored, and the implied ends of li/option/p/tr/td.
import { readFileSync } from 'node:fs';
import { resolve, dirname, basename } from 'node:path';
import { fileURLToPath } from 'node:url';
import { FakeDocument, FakeFragment, SVG_NS, HTML_NS, VOID_ELEMENTS } from './dom.js';

export const REPO_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..', '..');

const NAMED = {
  amp: '&', lt: '<', gt: '>', quot: '"', apos: "'", nbsp: ' ', copy: '©', reg: '®', hellip: '…', mdash: '—',
  ndash: '–', rsquo: '’', lsquo: '‘', ldquo: '“', rdquo: '”', times: '×', middot: '·', larr: '←', rarr: '→',
  uarr: '↑', darr: '↓', bull: '•', minus: '−', laquo: '«', raquo: '»', deg: '°', thinsp: ' ', ensp: ' ',
  emsp: ' ', zwj: '‍', zwnj: '‌', check: '✓', star: '☆', trade: '™', sect: '§', para: '¶',
  frac12: '½', plusmn: '±', le: '≤', ge: '≥', ne: '≠', lrm: '‎', rlm: '‏',
};

export function decodeEntities(text) {
  return String(text).replace(/&(#x[0-9a-fA-F]+|#\d+|[a-zA-Z][a-zA-Z0-9]*);/g, (all, body) => {
    if (body[0] === '#') {
      const code = body[1] === 'x' || body[1] === 'X' ? parseInt(body.slice(2), 16) : parseInt(body.slice(1), 10);
      return Number.isFinite(code) && code > 0 && code <= 0x10ffff ? String.fromCodePoint(code) : all;
    }
    return Object.prototype.hasOwnProperty.call(NAMED, body) ? NAMED[body] : all;
  });
}

const RAW = new Set(['script', 'style']);
const RCDATA = new Set(['textarea', 'title']);
const IMPLIED_END = {
  li: ['li'], option: ['option'], optgroup: ['optgroup', 'option'], dt: ['dt', 'dd'], dd: ['dt', 'dd'],
  tr: ['tr', 'td', 'th'], td: ['td', 'th'], th: ['td', 'th'], p: ['p'],
};
const P_CLOSERS = new Set(['address', 'article', 'aside', 'blockquote', 'details', 'dialog', 'div', 'dl', 'fieldset', 'figcaption', 'figure', 'footer', 'form', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'header', 'hr', 'main', 'menu', 'nav', 'ol', 'p', 'pre', 'section', 'table', 'ul']);

function parseAttributes(src) {
  const attrs = [];
  const re = /([^\s"'<>/=]+)(?:\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'=<>`]+)))?/g;
  let m;
  while ((m = re.exec(src)) !== null) {
    const value = m[2] ?? m[3] ?? m[4];
    attrs.push([m[1], value === undefined ? '' : decodeEntities(value)]);
  }
  return attrs;
}

/**
 * Parse markup and append the nodes to `parent` (a FakeElement, fragment or document).
 * @returns {object} parent
 */
export function parseInto(parent, html) {
  const doc = parent.nodeType === 9 ? parent : parent.ownerDocument;
  const stack = [parent];
  const top = () => stack[stack.length - 1];
  const inSvg = () => stack.some((n) => n.nodeType === 1 && n.namespaceURI === SVG_NS);
  const s = String(html);
  let i = 0;
  const appendText = (text) => {
    if (!text) return;
    const node = top();
    const last = node.childNodes[node.childNodes.length - 1];
    if (last && last.nodeType === 3) last.data += text;
    else node.appendChild(doc.createTextNode(text));
  };
  while (i < s.length) {
    const lt = s.indexOf('<', i);
    if (lt < 0) { appendText(decodeEntities(s.slice(i))); break; }
    if (lt > i) appendText(decodeEntities(s.slice(i, lt)));
    if (s.startsWith('<!--', lt)) {
      const end = s.indexOf('-->', lt + 4);
      i = end < 0 ? s.length : end + 3;
      continue;
    }
    if (s[lt + 1] === '!' || s[lt + 1] === '?') {
      const end = s.indexOf('>', lt);
      i = end < 0 ? s.length : end + 1;
      continue;
    }
    if (s[lt + 1] === '/') {
      const end = s.indexOf('>', lt);
      const name = s.slice(lt + 2, end < 0 ? s.length : end).trim();
      i = end < 0 ? s.length : end + 1;
      const lname = name.toLowerCase();
      for (let k = stack.length - 1; k >= 1; k -= 1) {
        const node = stack[k];
        if (node.localName === name || node.localName.toLowerCase() === lname) { stack.length = k; break; }
      }
      continue;
    }
    const m = /^<([a-zA-Z][a-zA-Z0-9:-]*)((?:\s+[^\s"'<>/=]+(?:\s*=\s*(?:"[^"]*"|'[^']*'|[^\s"'=<>`]+))?)*)\s*(\/?)>/.exec(s.slice(lt));
    if (!m) { appendText('<'); i = lt + 1; continue; }
    i = lt + m[0].length;
    const rawName = m[1];
    const svg = inSvg() || rawName.toLowerCase() === 'svg';
    const name = svg ? rawName : rawName.toLowerCase();
    if (!svg && IMPLIED_END[name]) {
      const closes = IMPLIED_END[name];
      while (stack.length > 1 && closes.includes(top().localName)) stack.pop();
    } else if (!svg && P_CLOSERS.has(name) && top().localName === 'p') {
      stack.pop();
    }
    const el = svg ? doc.createElementNS(SVG_NS, name) : doc.createElement(name);
    for (const [k, v] of parseAttributes(m[2])) {
      if (!el.hasAttribute(k)) el.setAttribute(svg ? k : k.toLowerCase(), v);
    }
    top().appendChild(el);
    const selfClosing = m[3] === '/' && (svg || el.namespaceURI !== HTML_NS);
    if ((el.namespaceURI === HTML_NS && VOID_ELEMENTS.has(name)) || selfClosing) continue;
    if (!svg && (RAW.has(name) || RCDATA.has(name))) {
      const close = new RegExp(`</${name}\\s*>`, 'i');
      const rest = s.slice(i);
      const found = close.exec(rest);
      const body = found ? rest.slice(0, found.index) : rest;
      if (body) el.appendChild(doc.createTextNode(RCDATA.has(name) ? decodeEntities(body) : body));
      i += found ? found.index + found[0].length : rest.length;
      continue;
    }
    stack.push(el);
  }
  return parent;
}

/**
 * Parse an HTML fragment.
 * @param {string} html
 * @param {object} [doc] the owner document (a new FakeDocument by default)
 * @returns {object[]} the top-level nodes (elements and text)
 */
export function parseFragment(html, doc = new FakeDocument()) {
  const frag = new FakeFragment(doc);
  parseInto(frag, html);
  return frag.childNodes.slice();
}

/**
 * Load an HTML file (relative to the repo root) into a new FakeDocument. A full document fills head and
 * body; a fragment goes into the body. Partials under src/static/partials/ are mounted inside the view
 * section they belong to (`<section class="view" id="view-<name>" data-view="<name>">`), slot partials
 * inside `<div id="<name>-slot" data-slot="<name>">`.
 * @param {string} file
 * @param {{wrap?: boolean}} [options]
 * @returns {FakeDocument}
 */
export function loadHtml(file, { wrap = true } = {}) {
  const path = resolve(REPO_ROOT, file);
  const text = readFileSync(path, 'utf8');
  const doc = new FakeDocument();
  if (/<html[\s>]/i.test(text)) {
    const tmp = new FakeFragment(doc);
    parseInto(tmp, text);
    const html = tmp.childNodes.find((n) => n.nodeType === 1 && n.localName === 'html');
    if (html) {
      doc.replaceChildren(html);
      if (!doc.head) html.insertBefore(doc.createElement('head'), html.firstChild);
      if (!doc.body) html.appendChild(doc.createElement('body'));
      // stray nodes around head/body go into body
      for (const child of html.childNodes.slice()) {
        if (child.nodeType === 1 && (child.localName === 'head' || child.localName === 'body')) continue;
        if (child.nodeType === 3 && !child.data.trim()) continue;
        doc.body.appendChild(child);
      }
    }
    return doc;
  }
  let host = doc.body;
  const isPartial = /[\\/]partials[\\/]/.test(path);
  if (wrap && isPartial) {
    const name = basename(path).replace(/\.html$/, '');
    const slots = { 'reader-player': ['rd-player-slot', 'reader-player'], 'speak-live': ['sp-live-slot', 'speak-live'] };
    if (slots[name]) {
      host = doc.createElement('div');
      host.setAttribute('id', slots[name][0]);
      host.setAttribute('data-slot', slots[name][1]);
    } else {
      host = doc.createElement('section');
      host.setAttribute('class', 'view');
      host.setAttribute('id', `view-${name}`);
      host.setAttribute('data-view', name);
    }
    doc.body.appendChild(host);
  }
  parseInto(host, text);
  return doc;
}
