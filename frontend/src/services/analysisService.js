import { analyzeIdea } from "../data/mockData";

/**
 * Action: Analyze current input text.
 * Simulates active dialectic processing and returns the result.
 */
export const performAnalysis = async (ideaText, selectedMode) => {
  if (!ideaText.trim()) return null;

  return new Promise((resolve, reject) => {
    // Simulate standard AI review latency
    setTimeout(() => {
      try {
        const result = analyzeIdea(ideaText, selectedMode);
        resolve(result);
      } catch (error) {
        console.error("Dialectic stress error on execution:", error);
        reject(error);
      }
    }, 1200);
  });
};
