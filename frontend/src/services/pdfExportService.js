import { jsPDF } from "jspdf";

/**
 * PDF Assessment exporter using jsPDF
 * Outputs a beautifully structured, minimalist executive review document.
 */
export const exportAssessmentToPDF = (activeAnalysis) => {
  if (!activeAnalysis) return;

  const { topic, score, risk, idea, mode, assumptions, counterArguments, verdict, date } = activeAnalysis;
  const formattedDate = date || new Date().toISOString().split("T")[0];
  const safeTopicName = (topic || "report").toLowerCase().replace(/[^a-z0-9]/gi, "_");

  const doc = new jsPDF({
    orientation: "portrait",
    unit: "mm",
    format: "a4"
  });

  // Dark grey primary typography, slate-500 secondary, safe borders
  const primaryColor = [15, 23, 42]; // slate-900
  const mutedColor = [100, 116, 139]; // slate-500
  const lightLine = [226, 232, 240]; // slate-200
  const lightFill = [248, 250, 252]; // slate-50

  // Coordinates and page borders
  let y = 20;
  const margin = 20;
  const pageWidth = doc.internal.pageSize.getWidth();
  const pageHeight = doc.internal.pageSize.getHeight();
  const contentWidth = pageWidth - margin * 2;

  // Helper: Dynamic page overflow safety
  const checkPageBreak = (neededHeight) => {
    if (y + neededHeight > pageHeight - margin) {
      doc.addPage();
      y = margin;
      return true;
    }
    return false;
  };

  // --- BRAND HEADER ---
  doc.setFont("Helvetica", "bold");
  doc.setFontSize(22);
  doc.setTextColor(primaryColor[0], primaryColor[1], primaryColor[2]);
  doc.text("ADVOCATE", margin, y);
  
  doc.setFont("Helvetica", "normal");
  doc.setFontSize(9.5);
  doc.setTextColor(mutedColor[0], mutedColor[1], mutedColor[2]);
  doc.text("/ Dialectical Assessment Report", margin + 50, y - 1);

  y += 4;
  doc.setDrawColor(lightLine[0], lightLine[1], lightLine[2]);
  doc.setLineWidth(0.5);
  doc.line(margin, y, margin + contentWidth, y);
  
  y += 10;

  // --- SUB HEADER INTRO ---
  doc.setFont("Helvetica", "bold");
  doc.setFontSize(9.5);
  doc.setTextColor(primaryColor[0], primaryColor[1], primaryColor[2]);
  doc.text("DIALECTICAL DIAGNOSTIC SURVEY", margin, y);
  
  doc.setFont("Helvetica", "normal");
  doc.setFontSize(9);
  doc.setTextColor(mutedColor[0], mutedColor[1], mutedColor[2]);
  doc.text(`DATE OF ISSUE: ${formattedDate}`, margin + contentWidth - 55, y);

  y += 6;

  // Stat Dashboard container block
  doc.setFillColor(lightFill[0], lightFill[1], lightFill[2]);
  doc.rect(margin, y, contentWidth, 20, "F");
  doc.rect(margin, y, contentWidth, 20, "S");

  doc.setFont("Helvetica", "bold");
  doc.setFontSize(8.5);
  doc.setTextColor(mutedColor[0], mutedColor[1], mutedColor[2]);
  doc.text("TOPIC DOMAIN", margin + 6, y + 7);
  doc.text("METHODOLOGY MODE", margin + 55, y + 7);
  doc.text("DIALECTIC RISK", margin + 105, y + 7);
  doc.text("STRESS SCORE", margin + 145, y + 7);

  doc.setFont("Helvetica", "bold");
  doc.setFontSize(9);
  doc.setTextColor(primaryColor[0], primaryColor[1], primaryColor[2]);
  
  const truncateTopic = (topic || "General Idea").toUpperCase();
  doc.text(truncateTopic.length > 22 ? truncateTopic.substring(0, 21) + "..." : truncateTopic, margin + 6, y + 14);
  doc.text(`${mode.toUpperCase()} MODE`, margin + 55, y + 14);
  
  // Risk Level Highlight
  let rColor = [100, 116, 139]; 
  if (risk.toLowerCase() === "high") rColor = [220, 38, 38]; 
  else if (risk.toLowerCase() === "medium") rColor = [217, 119, 6]; 
  else if (risk.toLowerCase() === "low") rColor = [5, 150, 105]; 
  
  doc.setTextColor(rColor[0], rColor[1], rColor[2]);
  doc.text(`${risk.toUpperCase()}`, margin + 105, y + 14);

  doc.setTextColor(primaryColor[0], primaryColor[1], primaryColor[2]);
  doc.text(`${score} / 100`, margin + 145, y + 14);

  y += 28;

  // --- SECTION 1: PROPOSAL ---
  checkPageBreak(30);
  doc.setFont("Helvetica", "bold");
  doc.setFontSize(10.5);
  doc.setTextColor(primaryColor[0], primaryColor[1], primaryColor[2]);
  doc.text("1. THE ORIGINAL PROPOSAL", margin, y);
  
  y += 5;
  doc.setFont("Helvetica", "normal");
  doc.setFontSize(10);
  doc.setTextColor(51, 65, 85);
  
  const splitProposal = doc.splitTextToSize(`"${idea}"`, contentWidth);
  doc.text(splitProposal, margin, y);
  y += (splitProposal.length * 5) + 10;

  // --- SECTION 2: HIDDEN ASSUMPTIONS ---
  checkPageBreak(30);
  doc.setFont("Helvetica", "bold");
  doc.setFontSize(10.5);
  doc.setTextColor(primaryColor[0], primaryColor[1], primaryColor[2]);
  doc.text("2. CONCEALED ARCHITECTURAL ASSUMPTIONS", margin, y);
  
  y += 5;
  doc.setFont("Helvetica", "normal");
  doc.setFontSize(9);
  doc.setTextColor(mutedColor[0], mutedColor[1], mutedColor[2]);
  doc.text("To succeed, this proposal is highly reliant on the following unproven assumptions remaining absolute truths:", margin, y);
  y += 6;

  doc.setFontSize(9.5);
  doc.setTextColor(primaryColor[0], primaryColor[1], primaryColor[2]);
  assumptions.forEach((assumption) => {
    const splitBullet = doc.splitTextToSize(`↳  ${assumption}`, contentWidth - 4);
    checkPageBreak(splitBullet.length * 5 + 4);
    doc.text(splitBullet, margin, y);
    y += (splitBullet.length * 5) + 3;
  });

  y += 8;

  // --- SECTION 3: COUNTER ARGUMENTS ---
  checkPageBreak(35);
  doc.setFont("Helvetica", "bold");
  doc.setFontSize(10.5);
  doc.setTextColor(primaryColor[0], primaryColor[1], primaryColor[2]);
  doc.text("3. SYSTEMIC COUNTER ARGUMENTS", margin, y);
  y += 6;

  counterArguments.forEach((argument) => {
    const colonIndex = argument.indexOf(":");
    let label = "";
    let bodyText = argument;

    if (colonIndex !== -1) {
      label = argument.substring(0, colonIndex + 1);
      bodyText = argument.substring(colonIndex + 1).trim();
    }

    const splitLabel = label ? doc.splitTextToSize(label, contentWidth - 8) : [];
    const splitBody = doc.splitTextToSize(label ? bodyText : argument, contentWidth - 8);
    const linesCount = splitLabel.length + splitBody.length;
    
    const cardHeight = (linesCount * 5) + 6;
    checkPageBreak(cardHeight + 6);

    // Simple light colored rect container
    doc.setFillColor(250, 250, 250);
    doc.setDrawColor(241, 245, 249);
    doc.setLineWidth(0.3);
    doc.rect(margin, y, contentWidth, cardHeight, "FD");

    let textY = y + 4.5;
    if (label) {
      doc.setFont("Helvetica", "bold");
      doc.setTextColor(primaryColor[0], primaryColor[1], primaryColor[2]);
      doc.text(splitLabel, margin + 4, textY);
      textY += (splitLabel.length * 5);
    }
    
    doc.setFont("Helvetica", "normal");
    doc.setTextColor(51, 65, 85);
    doc.text(splitBody, margin + 4, textY);
    
    y += cardHeight + 5;
  });

  y += 5;

  // --- SECTION 4: DIALECTIC VERDICT ---
  checkPageBreak(35);
  doc.setFont("Helvetica", "bold");
  doc.setFontSize(10.5);
  doc.setTextColor(primaryColor[0], primaryColor[1], primaryColor[2]);
  doc.text("4. DIALECTIC VERDICT", margin, y);
  y += 5;

  const splitVerdict = doc.splitTextToSize(`"${verdict}"`, contentWidth - 8);
  const boxHeight = (splitVerdict.length * 5) + 10;
  
  checkPageBreak(boxHeight + 10);

  // Render clean high-contrast card
  doc.setFillColor(248, 250, 252);
  doc.setDrawColor(15, 23, 42); // slate-900 border
  doc.setLineWidth(0.5);
  doc.rect(margin, y, contentWidth, boxHeight, "FD");

  doc.setFont("Helvetica", "bolditalic");
  doc.setFontSize(10);
  doc.setTextColor(primaryColor[0], primaryColor[1], primaryColor[2]);
  doc.text(splitVerdict, margin + 4, y + 7);

  // Document footer note
  y = pageHeight - 15;
  doc.setFont("Helvetica", "normal");
  doc.setFontSize(7.5);
  doc.setTextColor(mutedColor[0], mutedColor[1], mutedColor[2]);
  doc.text("ADVOCATE REPORT GENERATION SERVICE • STRUCTURAL VALIDITY UNIT • SYSTEM CONFIDENTIAL", margin, y);

  doc.save(`ADVOCATE-Assessment-${safeTopicName}.pdf`);
};
