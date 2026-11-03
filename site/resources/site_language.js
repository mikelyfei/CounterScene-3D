function translateText(text, locale) {
  const trimmed = text.trim();
  if (!trimmed) return text;
  if (locale.exact[trimmed]) return text.replace(trimmed, locale.exact[trimmed]);
  let value = text;
  for (const [source, target] of locale.phrases) value = value.replaceAll(source, target);
  return value;
}
if (typeof document !== 'undefined') {
  const selector = document.getElementById('language');
  const originals = new WeakMap();
  let locale, busy = false, scheduled = false;
  const observer = new MutationObserver(() => {
    if (!scheduled && !busy) { scheduled = true; queueMicrotask(() => {scheduled = false; applyLanguage();}); }
  });
  function applyLanguage() {
    if (selector.value === 'zh' && !locale) return;
    busy = true; observer.disconnect();
    const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
    while (walker.nextNode()) {
      const node = walker.currentNode;
      if (node.parentElement?.closest('script,style,pre,code')) continue;
      if (!originals.has(node)) originals.set(node, node.nodeValue);
      node.nodeValue = selector.value === 'zh' ? translateText(originals.get(node), locale) : originals.get(node);
    }
    document.documentElement.lang = selector.value === 'zh' ? 'zh-CN' : 'en';
    for(const link of document.querySelectorAll('[data-lang-link]')){const next=new URL(link.getAttribute('href'),location.href);next.searchParams.set('lang',selector.value);link.setAttribute('href',next.pathname.split('/').pop()+next.search+next.hash);}
    const frame = document.getElementById('reportFrame');
    const wanted = selector.value === 'zh' ? 'downloads/report_zh.pdf' : 'downloads/report.pdf';
    if (frame && frame.getAttribute('src') !== wanted) frame.setAttribute('src', wanted);
    busy = false; observer.observe(document.body, {subtree:true,childList:true,characterData:true});
  }
  async function changeLanguage() {
    if (selector.value === 'zh' && !locale) {
      const response = await fetch('locales/zh.json?v=annotations1');
      if (!response.ok) throw Error('Chinese locale unavailable');
      locale = await response.json();
    }
    const url = new URL(location.href);url.searchParams.set('lang', selector.value);history.replaceState(null,'',url);
    applyLanguage();
  }
  selector.value = new URL(location.href).searchParams.get('lang') === 'zh' ? 'zh' : 'en';
  selector.addEventListener('change', () => changeLanguage().catch(error => {document.getElementById('error').textContent=error.message;}));
  changeLanguage();
}
