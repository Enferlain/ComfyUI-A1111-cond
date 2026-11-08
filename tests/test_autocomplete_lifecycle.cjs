// Run with: node --test tests/test_autocomplete_lifecycle.cjs
// Exercise the extension's real handlers using a minimal DOM and controlled I/O.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const { test } = require("node:test");

class Element {
  constructor() {
    this.style = {};
    this.children = [];
    this.listeners = new Map();
    this.isConnected = true;
    this.classList = { add() {}, remove() {} };
  }
  addEventListener(name, handler) {
    if (!this.listeners.has(name)) this.listeners.set(name, new Set());
    this.listeners.get(name).add(handler);
  }
  removeEventListener(name, handler) {
    this.listeners.get(name)?.delete(handler);
  }
  dispatchEvent(event) {
    event.target ??= this;
    for (const handler of this.listeners.get(event.type) || []) handler(event);
  }
  appendChild(child) {
    this.children.push(child);
    child.parentNode = this;
  }
  set innerHTML(value) {
    this.html = value;
    this.children = [];
  }
  get innerHTML() { return this.html || ""; }
  contains(target) {
    return target === this || this.children.some(child => child.contains(target));
  }
  querySelectorAll() { return this.children; }
  scrollIntoView() {}
}

function harness() {
  const timers = new Map();
  const frames = new Map();
  const requests = [];
  let nextId = 1;
  const document = new Element();
  Object.assign(document, {
    head: new Element(), body: new Element(), activeElement: null,
    createElement: () => new Element(),
  });
  const context = vm.createContext({
    console: { warn() {} }, document,
    window: { innerHeight: 1000, innerWidth: 1000 },
    localStorage: { getItem() { return null; }, setItem() {} },
    Event: class { constructor(type) { this.type = type; } },
    app: { registerExtension(extension) { context.extension = extension; } },
    setTimeout(fn) { const id = nextId++; timers.set(id, fn); return id; },
    clearTimeout(id) { timers.delete(id); },
    requestAnimationFrame(fn) { const id = nextId++; frames.set(id, fn); return id; },
    cancelAnimationFrame(id) { frames.delete(id); },
    fetch(url, options) {
      if (url.endsWith("/warmup")) return Promise.resolve({ ok: true });
      return new Promise((resolve, reject) => requests.push({
        body: JSON.parse(options.body), reject,
        resolve(results) { resolve({ ok: true, json: async () => ({ results }) }); },
      }));
    },
  });
  const source = fs.readFileSync(path.join(__dirname, "../js/autocomplete.js"), "utf8")
    .replace(/^import .*;$/m, "").replace(/^export /gm, "");
  vm.runInContext(source, context);
  // Layout is a browser concern; retain real popup rendering and event handlers.
  vm.runInContext("getCaretCoordinates = () => ({x: 10, y: 10, height: 20})", context);
  const state = () => vm.runInContext(`({
    owner: currentTextarea, names: currentResults.map(tag => tag.name),
    popup: autocompletePopup, selected: selectedIndex
  })`, context);
  const flush = async () => { for (let i = 0; i < 10; i++) await Promise.resolve(); };
  const fireTimer = () => {
    const entry = timers.entries().next().value;
    assert.ok(entry, "expected a scheduled timer");
    timers.delete(entry[0]);
    return entry[1]();
  };
  async function addNode(value = "ca") {
    const textarea = new Element();
    Object.assign(textarea, { value, selectionStart: value.length, selectionEnd: value.length });
    textarea.focus = () => {
      if (document.activeElement === textarea) return;
      document.activeElement?.dispatchEvent({ type: "blur" });
      document.activeElement = textarea;
      textarea.dispatchEvent({ type: "focus" });
    };
    textarea.setSelectionRange = (start, end) => {
      textarea.selectionStart = start; textarea.selectionEnd = end;
    };
    const widget = { name: "text", inputEl: textarea };
    const node = { comfyClass: "A1111Prompt", widgets: [widget] };
    await context.extension.nodeCreated(node);
    const pendingFrames = [...frames.values()]; frames.clear();
    pendingFrames.forEach(fn => fn());
    return { textarea, node, widget };
  }
  async function start(textarea) {
    textarea.focus();
    textarea.dispatchEvent({ type: "input" });
    fireTimer();
    await flush();
  }
  function key(textarea, key) {
    const event = { type: "keydown", key, prevented: false,
      preventDefault() { this.prevented = true; } };
    textarea.dispatchEvent(event);
    return event;
  }
  return { context, document, timers, frames, requests, state, flush, fireTimer, addNode, start, key };
}

test("late responses cannot replace another focused node's suggestions", async () => {
  const h = harness();
  const a = await h.addNode(); const b = await h.addNode("do");
  await h.start(a.textarea);
  b.textarea.focus(); h.fireTimer();
  await h.start(b.textarea);
  h.requests[1].resolve([{ name: "dog", count: 1 }]); await h.flush();
  h.requests[0].resolve([{ name: "cat", count: 1 }]); await h.flush();
  assert.equal(h.state().owner, b.textarea);
  assert.deepEqual(Array.from(h.state().names), ["dog"]);
});

test("late errors cannot erase another node's results", async () => {
  const h = harness();
  const a = await h.addNode(); const b = await h.addNode("do");
  await h.start(a.textarea);
  b.textarea.focus(); h.fireTimer();
  await h.start(b.textarea);
  h.requests[1].resolve([{ name: "dog", count: 1 }]); await h.flush();
  h.requests[0].reject(new Error("offline")); await h.flush();
  assert.equal(h.state().owner, b.textarea);
  assert.deepEqual(Array.from(h.state().names), ["dog"]);
});

test("other nodes' click handlers and blur timers preserve the current popup", async () => {
  const h = harness();
  const a = await h.addNode(); const b = await h.addNode("do");
  await h.start(a.textarea);
  h.requests[0].resolve([{ name: "cat", count: 1 }]); await h.flush();
  b.textarea.focus();
  // Schedule B while retaining A's delayed blur callback.
  b.textarea.dispatchEvent({ type: "input" });
  const blur = h.timers.get(a.textarea._a1111_autocomplete_blur_timeout);
  h.timers.delete(a.textarea._a1111_autocomplete_blur_timeout);
  h.fireTimer(); await h.flush();
  h.requests[1].resolve([{ name: "dog", count: 1 }]); await h.flush();
  h.document.dispatchEvent({ type: "click", target: b.textarea });
  blur();
  assert.equal(h.state().owner, b.textarea);
  assert.equal(h.state().popup.style.display, "block");
});

test("blur and Escape invalidate in-flight requests", async () => {
  for (const dismiss of ["blur", "Escape"]) {
    const h = harness(); const a = await h.addNode();
    await h.start(a.textarea);
    if (dismiss === "blur") {
      a.textarea.dispatchEvent({ type: "blur" }); h.document.activeElement = null;
    } else h.key(a.textarea, "Escape");
    h.requests[0].resolve([{ name: "cat", count: 1 }]); await h.flush();
    assert.equal(h.state().owner, null);
  }
});

test("current errors are visible, dismissible, and recover on another query", async () => {
  const h = harness(); const a = await h.addNode();
  await h.start(a.textarea);
  h.requests[0].reject(new Error("offline")); await h.flush();
  assert.equal(h.state().owner, a.textarea);
  assert.match(h.state().popup.innerHTML, /Failed to fetch/);
  assert.equal(h.key(a.textarea, "ArrowDown").prevented, false);
  assert.equal(h.state().selected, -1);
  h.key(a.textarea, "Escape");
  assert.equal(h.state().popup.style.display, "none");
  await h.start(a.textarea);
  h.requests[1].resolve([{ name: "cat", count: 1 }]); await h.flush();
  h.key(a.textarea, "ArrowDown"); h.key(a.textarea, "Enter");
  assert.equal(a.textarea.value, "cat, ");
});

test("popup click insertion survives blur and cleanup invalidates requests", async () => {
  const h = harness(); const a = await h.addNode();
  await h.start(a.textarea);
  h.requests[0].resolve([{ name: "cat", count: 1 }]); await h.flush();
  a.textarea.dispatchEvent({ type: "blur" }); h.document.activeElement = null;
  h.state().popup.children[0].dispatchEvent({ type: "click" });
  assert.equal(a.textarea.value, "cat, ");
  assert.equal(h.document.activeElement, a.textarea);
  a.textarea.value = "do"; a.textarea.selectionStart = 2;
  await h.start(a.textarea);
  a.node.onRemoved();
  h.requests[1].resolve([{ name: "dog", count: 1 }]); await h.flush();
  assert.equal(h.state().owner, null);
  assert.equal(h.frames.size, 0);
});

test("rebinding a replaced textarea discards its pending response", async () => {
  const h = harness(); const a = await h.addNode(); const b = await h.addNode("do");
  await h.start(a.textarea);
  a.textarea.isConnected = false;
  a.widget.inputEl = b.textarea;
  const pendingFrames = [...h.frames.values()]; h.frames.clear();
  pendingFrames.forEach(fn => fn());
  h.requests[0].resolve([{ name: "cat", count: 1 }]); await h.flush();
  assert.equal(h.state().owner, null);
  assert.equal(a.textarea.listeners.get("input").size, 0);
});

test("refocusing cancels old blur timers without reviving old requests", async () => {
  const h = harness(); const a = await h.addNode(); const b = await h.addNode();
  await h.start(a.textarea);
  b.textarea.focus();
  a.textarea.focus();
  h.requests[0].resolve([{ name: "cat", count: 1 }]); await h.flush();
  assert.equal(h.state().owner, null);
  assert.equal(a.textarea._a1111_autocomplete_blur_timeout, null);
});
