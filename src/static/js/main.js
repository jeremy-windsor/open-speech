/**
 * The entry module. Everything happens in shell/index.js boot(): this file only starts it, so the
 * boot sequence stays testable in Node (importing a module never has side effects, except here and in
 * theme-boot.js).
 */
import { boot } from './shell/index.js';

boot().then((ok) => {
  // Speak's voice for the Reader before Speak's view has run: published from Speak's saved setup
  // or its default once the model list is known. Imported after the first view mounted (not part of the
  // preloaded static graph), so it never delays the first view or anything on the way to first audio.
  if (!ok) return;
  import('./views/speak/speak-voice.js')
    .then((voice) => { voice.startBootVoice(); })
    .catch((error) => { if (globalThis.console) globalThis.console.error(error); });
});
