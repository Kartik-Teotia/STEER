const root = document.documentElement;
const header = document.querySelector("[data-header]");
const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

const updateScrollState = () => {
  const scrollable = document.documentElement.scrollHeight - window.innerHeight;
  const progress = scrollable > 0 ? window.scrollY / scrollable : 0;
  root.style.setProperty("--scroll", Math.min(1, Math.max(0, progress)));
  header?.classList.toggle("is-scrolled", window.scrollY > 24);
};

updateScrollState();
window.addEventListener("scroll", updateScrollState, { passive: true });

const revealItems = document.querySelectorAll(".reveal");
if (reducedMotion || !("IntersectionObserver" in window)) {
  revealItems.forEach((item) => item.classList.add("is-visible"));
} else {
  const revealObserver = new IntersectionObserver(
    (entries, observer) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("is-visible");
          observer.unobserve(entry.target);
        }
      });
    },
    { rootMargin: "0px 0px -8% 0px", threshold: 0.08 },
  );
  revealItems.forEach((item) => revealObserver.observe(item));
}

const ambientVideos = document.querySelectorAll(
  ".ambient-video:not([data-smart-video])",
);
if ("IntersectionObserver" in window) {
  const videoObserver = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        const video = entry.target;
        if (entry.isIntersecting) {
          video.play().catch(() => {});
        } else {
          video.pause();
        }
      });
    },
    { rootMargin: "160px 0px", threshold: 0.12 },
  );
  ambientVideos.forEach((video) => videoObserver.observe(video));
}

const smartVideos = [...document.querySelectorAll("[data-smart-video]")];

const soundButtonFor = (video) => {
  const scope = video.closest(
    ".hero-media, .speaker-example, .video-card, .role-card, .live-card",
  );
  return scope?.querySelector("[data-sound-toggle]");
};

const updateSoundButton = (video) => {
  const button = soundButtonFor(video);
  if (!button) return;
  const isPlayingWithSound =
    !video.paused && !video.muted && video.volume > 0;
  button.textContent = isPlayingWithSound ? "Pause" : "Play with sound";
  button.setAttribute("aria-pressed", String(isPlayingWithSound));
};

const stopOtherVideos = (activeVideo) => {
  smartVideos.forEach((video) => {
    if (video === activeVideo) return;
    if (!video.paused) video.pause();
    if (!video.muted) video.muted = true;
    updateSoundButton(video);
  });
};

smartVideos.forEach((video) => {
  const button = soundButtonFor(video);
  updateSoundButton(video);

  button?.addEventListener("click", () => {
    if (video.paused || video.muted || video.volume === 0) {
      stopOtherVideos(video);
      video.muted = false;
      video.volume = 1;
      video.play().catch(() => {});
    } else {
      video.pause();
      video.muted = true;
    }
    updateSoundButton(video);
  });

  video.addEventListener("play", () => {
    stopOtherVideos(video);
    video.muted = false;
    video.volume = 1;
    updateSoundButton(video);
  });

  video.addEventListener("pause", () => updateSoundButton(video));

  video.addEventListener("volumechange", () => {
    if (!video.paused && !video.muted && video.volume > 0) {
      stopOtherVideos(video);
    }
    updateSoundButton(video);
  });
});

if ("IntersectionObserver" in window) {
  const smartVideoObserver = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting && entry.intersectionRatio >= 0.2) return;
        const video = entry.target;
        video.pause();
        video.muted = true;
        updateSoundButton(video);
      });
    },
    { threshold: [0, 0.2] },
  );

  smartVideos.forEach((video) => smartVideoObserver.observe(video));
}
