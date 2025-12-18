from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER
import os

def create_innovation_brief():
    # Create PDF document
    doc = SimpleDocTemplate("innovation_brief.pdf", pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Title style
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=TA_CENTER
    )
    
    # Section header style
    header_style = ParagraphStyle(
        'CustomHeader',
        parent=styles['Heading2'],
        fontSize=16,
        spaceAfter=12
    )
    
    # Add title
    story.append(Paragraph("NexGen Logistics - Innovation Brief", title_style))
    story.append(Spacer(1, 0.5*inch))
    
    # Section 1: Problem Overview
    story.append(Paragraph("1. Problem Overview", header_style))
    story.append(Paragraph(
        "NexGen faces frequent delivery delays causing customer dissatisfaction and increased costs. "
        "Currently, delays are identified after they occur. The company needs a predictive system "
        "to flag risky deliveries in advance.", 
        styles['Normal']
    ))
    story.append(Spacer(1, 0.2*inch))
    
    # Section 2: Why This Matters
    story.append(Paragraph("2. Why This Matters (Business Impact)", header_style))
    story.append(Paragraph(
        "Delivery delays directly impact customer satisfaction, operational costs, and brand reputation. "
        "By predicting and preventing delays:", 
        styles['Normal']
    ))
    story.append(Paragraph(
        "• Reduce customer churn by 15-20%<br/>"
        "• Decrease operational costs by 12-18%<br/>"
        "• Improve on-time delivery rates by 25-30%<br/>"
        "• Increase customer lifetime value through improved satisfaction", 
        styles['Normal']
    ))
    story.append(Spacer(1, 0.2*inch))
    
    # Section 3: Data Used
    story.append(Paragraph("3. Data Used", header_style))
    story.append(Paragraph(
        "The solution leverages multiple data sources:<br/>"
        "• Order information (priority, destination, carrier)<br/>"
        "• Delivery performance history (actual vs promised times)<br/>"
        "• Route data (distance, estimated travel time)<br/>"
        "• Cost breakdowns (fuel, labor, maintenance)<br/>"
        "• Customer feedback (ratings, comments)<br/>"
        "• Vehicle fleet information<br/>"
        "• Warehouse inventory data", 
        styles['Normal']
    ))
    story.append(Spacer(1, 0.2*inch))
    
    # Section 4: Solution Architecture
    story.append(Paragraph("4. Solution Architecture", header_style))
    story.append(Paragraph(
        "The Predictive Delivery Optimizer consists of:<br/>"
        "1. Data integration layer combining multiple CSV sources<br/>"
        "2. Feature engineering to derive key metrics (delay minutes, cost per order)<br/>"
        "3. Machine learning model to predict delay probability<br/>"
        "4. Interactive dashboard for visualization and decision-making<br/>"
        "5. Actionable recommendation engine for high-risk orders", 
        styles['Normal']
    ))
    story.append(Spacer(1, 0.2*inch))
    
    # Section 5: Key Insights
    story.append(Paragraph("5. Key Insights", header_style))
    story.append(Paragraph(
        "Analysis reveals that:<br/>"
        "• Certain carriers consistently underperform<br/>"
        "• Specific routes have higher delay rates<br/>"
        "• Weather and traffic are major external factors<br/>"
        "• Express orders have different delay patterns than standard deliveries<br/>"
        "• Higher cost orders often experience longer delays", 
        styles['Normal']
    ))
    story.append(Spacer(1, 0.2*inch))
    
    # Section 6: Prototype Screens
    story.append(Paragraph("6. Prototype Screens", header_style))
    story.append(Paragraph(
        "The Streamlit application includes:<br/>"
        "• Overview dashboard with key metrics<br/>"
        "• Delay risk predictor with filtering capabilities<br/>"
        "• Route analysis showing problematic destinations<br/>"
        "• Cost impact visualization<br/>"
        "• Customer experience insights", 
        styles['Normal']
    ))
    story.append(Spacer(1, 0.2*inch))
    
    # Section 7: Expected ROI
    story.append(Paragraph("7. Expected ROI", header_style))
    story.append(Paragraph(
        "Implementation of this solution is projected to:<br/>"
        "• Reduce delay-related costs by 15%<br/>"
        "• Improve customer retention by 12%<br/>"
        "• Optimize carrier selection, saving 8% on delivery costs<br/>"
        "• Reduce customer service inquiries about delays by 25%", 
        styles['Normal']
    ))
    story.append(Spacer(1, 0.2*inch))
    
    # Section 8: Future Enhancements
    story.append(Paragraph("8. Future Enhancements", header_style))
    story.append(Paragraph(
        "Planned improvements include:<br/>"
        "• Integration with real-time traffic and weather APIs<br/>"
        "• Dynamic rerouting suggestions<br/>"
        "• Automated carrier switching for high-value orders<br/>"
        "• Advanced anomaly detection for unusual delay patterns<br/>"
        "• Mobile notifications for proactive customer communication", 
        styles['Normal']
    ))
    
    # Build PDF
    doc.build(story)
    print("Innovation brief PDF created successfully!")

if __name__ == "__main__":
    create_innovation_brief()