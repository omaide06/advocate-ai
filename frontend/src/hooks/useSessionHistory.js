import { useState } from "react";
import { performAnalysis } from "../services/analysisService";
import { initialHistory } from "../data/mockHistory";

export const useSessionHistory = () => {
  // Input fields state
  const [ideaText, setIdeaText] = useState("");
  const [selectedMode, setSelectedMode] = useState("Demolish");
  
  // Active report states
  const [activeAnalysis, setActiveAnalysis] = useState(null);
  const [activeSessionId, setActiveSessionId] = useState(null);
  
  // Loading overlay state
  const [isLoading, setIsLoading] = useState(false);

  // History state
  const [historyList, setHistoryList] = useState(initialHistory);

  /**
   * Action: Analyze current input text.
   * Simulates active dialectic processing, generates results, and writes to history.
   */
  const handleAnalyze = async () => {
    if (!ideaText.trim() || isLoading) return;

    setIsLoading(true);
    setActiveAnalysis(null);
    setActiveSessionId(null);

    try {
      const result = await performAnalysis(ideaText, selectedMode);
      
      if (result) {
        const newId = Date.now();
        const today = new Date().toISOString().split("T")[0];
        
        const newRecord = {
          id: newId,
          idea: ideaText.trim(),
          mode: selectedMode,
          risk: result.risk,
          score: result.score,
          date: today,
          topic: result.topic,
          assumptions: result.assumptions,
          counterArguments: result.counterArguments,
          verdict: result.verdict
        };

        // Store new evaluation both globally and at the front of table listings
        setActiveAnalysis(newRecord);
        setActiveSessionId(newId);
        setHistoryList((prev) => [newRecord, ...prev]);
      }
    } catch (error) {
      // Error is logged in the service, but we handle the loading state cleanup here
    } finally {
      setIsLoading(false);
    }
  };

  /**
   * Restore historical results back to viewport focus
   * 
   * @param {object} session - Session selection
   */
  const handleSelectSession = (session) => {
    setIdeaText(session.idea);
    setSelectedMode(session.mode);
    setActiveAnalysis(session);
    setActiveSessionId(session.id);
  };

  return {
    ideaText,
    setIdeaText,
    selectedMode,
    setSelectedMode,
    activeAnalysis,
    activeSessionId,
    isLoading,
    historyList,
    handleAnalyze,
    handleSelectSession
  };
};
