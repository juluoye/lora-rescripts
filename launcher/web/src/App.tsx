import React from 'react';
import { Sidebar } from './components/Sidebar';
import { Header } from './components/Header';
import { PageRouter } from './components/PageRouter';
import { LauncherBootScreen } from './components/LauncherBootScreen';
import { AppProvider, useApp } from './context/AppContext';
import { useTranslation } from './hooks/useTranslation';

function AppFooter() {
  const { language, translations } = useApp();
  const { t } = useTranslation(translations, language);

  return (
    <footer className="px-8 py-3 text-[10px] flex justify-end items-center" style={{ color: 'var(--text-dim)', borderTop: '1px solid var(--border)' }}>
      <a
        href="https://github.com/WhitecrowAurora/lora-rescripts"
        target="_blank"
        rel="noopener noreferrer"
        className="cursor-pointer hover:underline"
        style={{ color: 'var(--text-muted)' }}
      >
        {t('about_github')}
      </a>
    </footer>
  );
}

function AppLayout() {
  const { ready, bootstrapped, bootStage, bootError, language, version, theme, runtimeRecommendation } = useApp();

  return (
    <div className="relative h-screen w-full overflow-hidden" style={{ backgroundColor: 'var(--bg-base)', color: 'var(--text-primary)' }}>
      <div className={`flex h-screen w-full overflow-hidden font-sans select-none transition-opacity duration-300 ${bootstrapped ? 'opacity-100' : 'opacity-0'}`}>
        <Sidebar />
        <main className="flex-1 flex flex-col relative">
          {/* Decorative gradient blobs */}
          <div className="absolute top-0 right-0 w-[500px] h-[500px] blur-[120px] rounded-full -mr-64 -mt-64 pointer-events-none" style={{ backgroundColor: 'var(--accent)', opacity: 0.03 }} />
          <div className="absolute bottom-0 left-0 w-[300px] h-[300px] blur-[100px] rounded-full -ml-32 -mb-32 pointer-events-none" style={{ backgroundColor: 'var(--secondary)', opacity: 0.03 }} />

          {ready && <Header />}

          <section className="flex-1 overflow-y-auto p-8 pb-16 z-10 custom-scrollbar">
            <div className="max-w-4xl mx-auto h-full">
              {bootstrapped ? <PageRouter /> : null}
            </div>
          </section>

          {ready && <AppFooter />}
        </main>
      </div>

      <LauncherBootScreen
        visible={!bootstrapped}
        stage={bootStage}
        language={language}
        version={version}
        theme={theme}
        error={bootError}
        gpuVendor={runtimeRecommendation?.gpu_vendor}
      />
    </div>
  );
}

export default function App() {
  return (
    <AppProvider>
      <AppLayout />
    </AppProvider>
  );
}
