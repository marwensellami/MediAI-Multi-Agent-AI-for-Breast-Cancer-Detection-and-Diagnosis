// src/utils/downloadPDF.ts
import html2canvas from 'html2canvas';
import jsPDF from 'jspdf';

/**
 * Exporte un élément HTML en PDF sur une seule page continue (hauteur adaptée au contenu)
 * @param elementId - ID de l'élément à exporter
 * @param filename - Nom du fichier PDF (sans extension)
 * @param orientation - 'portrait' ou 'landscape'
 */
export async function downloadPDF(
  elementId: string,
  filename: string = 'rapport_medical',
  orientation: 'portrait' | 'landscape' = 'portrait'
): Promise<void> {
  const element = document.getElementById(elementId);
  if (!element) {
    console.error(`Élément avec l'ID "${elementId}" introuvable.`);
    return;
  }

  // Sauvegarder les styles originaux
  const originalHeight = element.style.height;
  const originalOverflow = element.style.overflow;
  const originalMaxHeight = element.style.maxHeight;

  // Désactiver les contraintes de hauteur pour capturer tout le contenu
  element.style.height = 'auto';
  element.style.overflow = 'visible';
  element.style.maxHeight = 'none';

  try {
    // Capturer l'élément en une seule image
    const canvas = await html2canvas(element, {
      scale: 2,
      useCORS: true,
      backgroundColor: '#ffffff',
      scrollY: 0,
      windowHeight: element.scrollHeight,
    });

    const pdf = new jsPDF({
      orientation,
      unit: 'mm',
      format: [210, 1000], // 📌 Largeur A4 (210mm) et hauteur très grande (1000mm)
    });

    const pdfWidth = 210;
    const margin = 10;
    const contentWidth = pdfWidth - 2 * margin;

    const imgWidth = canvas.width;
    const imgHeight = canvas.height;

    const ratio = contentWidth / imgWidth;
    const pdfContentHeight = imgHeight * ratio;

    // Ajuster la hauteur du PDF à la hauteur réelle du contenu
    const totalHeight = pdfContentHeight + 2 * margin;
    pdf.internal.pageSize.height = totalHeight;
    pdf.internal.pageSize.width = pdfWidth;

    pdf.addImage(canvas.toDataURL('image/png'), 'PNG', margin, margin, contentWidth, pdfContentHeight);

    pdf.save(`${filename}.pdf`);
  } catch (error) {
    console.error('Erreur lors de l\'export PDF :', error);
    throw error;
  } finally {
    // Restaurer les styles d'origine
    element.style.height = originalHeight;
    element.style.overflow = originalOverflow;
    element.style.maxHeight = originalMaxHeight;
  }
}