import React from "react";
import Header from "./components/Header";
import InputPanel from "./components/InputPanel";
import OutputPanel from "./components/OutputPanel";
import { useSessionHistory } from "./hooks/useSessionHistory";
import { exportAssessmentToPDF } from "./services/pdfExportService";

/**
 * ============================================================================
 * ADVOCATE App component
 * ============================================================================
 * LAYOUT STRUCTURE:
 * - A simple horizontal top banner showing Logo and PDF/TXT export trigger.
 * - Two-column main workspace:
 *   - Left Column: Active inputs (Idea textual entry, selected mode radios, analysis button).
 *   - Right Column: Immediate dialectical output panels (Quality scorecard, assumptions list, detailed rebuttals, verdict card).
 * - Bottom Row Full-Width: Historical Session Ledger listing past attempts with active recall support.
 * 
 * WHY TWO-COLUMN DESIGN IS USED:
 * - Emulates side-by-side editing workflows resembling professional editor/previewer or terminal-oriented debuggers.
 * - Ensures high visibility side-by-side relative comparison without vertical scrolling stress.
 */
export default function App() {
  const {
    ideaText,
    setIdeaText,
    selectedMode,
    setSelectedMode,
    activeAnalysis,
    isLoading,
    handleAnalyze
  } = useSessionHistory();

  const handleExportPDF = () => {
    exportAssessmentToPDF(activeAnalysis);
  };

  return (
    <div className="min-h-screen bg-[#fafafa] flex flex-col font-sans text-slate-800">
      {/* 1️⃣ Top Horizonal Header */}
      <Header 
        hasAnalysis={!!activeAnalysis && !isLoading} 
        onExport={handleExportPDF} 
      />

      {/* Main Container Workspace */}
      <main className="flex-grow max-w-7xl w-full mx-auto p-6 md:p-8 space-y-8">
        
        {/* 2️⃣ Workstation Main View: 2-Column Side-by-Side Flex */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
          
          {/* Left Column Area (Span 5 on Large Screens) */}
          <div className="lg:col-span-5 h-full">
            <InputPanel
              ideaText={ideaText}
              setIdeaText={setIdeaText}
              selectedMode={selectedMode}
              setSelectedMode={setSelectedMode}
              onAnalyze={handleAnalyze}
              isLoading={isLoading}
            />
          </div>

          {/* Right Column Area (Span 7 on Large Screens) */}
          <div className="lg:col-span-7 h-full">
            <OutputPanel
              analysisResult={activeAnalysis}
              isLoading={isLoading}
            />
          </div>

        </div>

      </main>

      {/* Corporate aesthetic system-level footer */}
      <footer className="py-6 border-t border-gray-200 bg-white text-center select-none text-[10px] text-gray-400 font-sans tracking-wide">
        ADVOCATE DIALECTICAL OFFICE SYSTEM © 2026 • WORK WITHIN TOTAL REASON
      </footer>
    </div>
  );
}

