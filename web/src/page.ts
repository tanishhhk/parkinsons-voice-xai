// Static page furniture: the cohort grid and the section navigation state.

/** One mark per person: filled squares for Parkinson's, rings for healthy. */
export function drawCohort(el: HTMLElement, pd: number, healthy: number): void {
  const marks: HTMLElement[] = [];
  for (let i = 0; i < pd + healthy; i++) {
    const m = document.createElement("i");
    m.className = i < pd ? "pd" : "hc";
    marks.push(m);
  }
  el.setAttribute("role", "img");
  el.setAttribute("aria-label", `${pd} people with Parkinson's and ${healthy} healthy people`);
  el.replaceChildren(...marks);
}

/** Mark the nav link of the section currently in view. */
export function trackSections(nav: HTMLElement): void {
  const links = new Map<string, HTMLAnchorElement>();
  nav.querySelectorAll<HTMLAnchorElement>("a[href^='#']").forEach((a) => links.set(a.hash.slice(1), a));
  const observer = new IntersectionObserver(
    (entries) => {
      for (const e of entries) {
        if (!e.isIntersecting) continue;
        links.forEach((a) => a.removeAttribute("aria-current"));
        const link = links.get(e.target.id);
        if (link) {
          link.setAttribute("aria-current", "true");
          const list = link.closest("ol");
          if (list) list.scrollLeft = link.offsetLeft - (list.clientWidth - link.offsetWidth) / 2;
        }
      }
    },
    { rootMargin: "-40% 0px -55% 0px" },
  );
  links.forEach((_, id) => {
    const section = document.getElementById(id);
    if (section) observer.observe(section);
  });
}
