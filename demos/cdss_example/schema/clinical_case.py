from pydantic import BaseModel, Field
from typing import Optional, List
class VitalSigns(BaseModel):
    """Patient vital signs"""
    systolic_bp: Optional[int] = Field(None, description="Systolic blood pressure (mmHg)")
    diastolic_bp: Optional[int] = Field(None, description="Diastolic blood pressure (mmHg)")
    heart_rate: Optional[int] = Field(None, description="Heart rate (bpm)")
    respiratory_rate: Optional[int] = Field(None, description="Respiratory rate (per minute)")
    temperature: Optional[float] = Field(None, description="Temperature (°C or °F)")
    oxygen_saturation: Optional[float] = Field(None, description="Oxygen saturation (%)")
    weight: Optional[float] = Field(None, description="Weight (kg)")
    height: Optional[float] = Field(None, description="Height (cm)")


class LabResult(BaseModel):
    """Laboratory test result"""
    test_name: str = Field(description="Name of the laboratory test")
    value: Optional[float] = Field(None, description="Test value")
    unit: Optional[str] = Field(None, description="Unit of measurement")
    reference_range: Optional[str] = Field(None, description="Normal reference range")
    status: Optional[str] = Field(None, description="Status: normal, high, low, critical")
    date: Optional[str] = Field(None, description="Date of test")


class ClinicalCase(BaseModel):
    """Clinical case input data model"""
    patient_id: Optional[str] = Field(None, description="Patient identifier")
    age: Optional[int] = Field(None, description="Patient age in years")
    sex: Optional[str] = Field(None, description="Patient sex (M/F/Other)")
    chief_complaint: Optional[str] = Field(None, description="Chief complaint")
    history_of_present_illness: Optional[str] = Field(None, description="History of present illness")
    past_medical_history: Optional[List[str]] = Field(default_factory=list, description="Past medical conditions")
    medications: Optional[List[str]] = Field(default_factory=list, description="Current medications")
    allergies: Optional[List[str]] = Field(default_factory=list, description="Known allergies")
    vital_signs: Optional[VitalSigns] = Field(None, description="Current vital signs")
    lab_results: Optional[List[LabResult]] = Field(default_factory=list, description="Laboratory results")
    imaging_results: Optional[List[str]] = Field(default_factory=list, description="Imaging study results")
    physical_exam: Optional[str] = Field(None, description="Physical examination findings")
    free_text: Optional[str] = Field(None, description="Free-text clinical notes or case description")
    
    def to_clinical_summary(self) -> str:
        """Convert structured case to clinical summary text"""
        parts = []
        
        if self.patient_id:
            parts.append(f"Patient ID: {self.patient_id}")
        if self.age and self.sex:
            parts.append(f"Patient: {self.age}-year-old {self.sex}")
        elif self.age:
            parts.append(f"Age: {self.age} years")
        
        if self.chief_complaint:
            parts.append(f"\nChief Complaint: {self.chief_complaint}")
        
        if self.history_of_present_illness:
            parts.append(f"\nHistory of Present Illness:\n{self.history_of_present_illness}")
        
        if self.past_medical_history:
            parts.append(f"\nPast Medical History: {', '.join(self.past_medical_history)}")
        
        if self.medications:
            parts.append(f"\nMedications: {', '.join(self.medications)}")
        
        if self.allergies:
            parts.append(f"\nAllergies: {', '.join(self.allergies)}")
        
        if self.vital_signs:
            vitals = []
            if self.vital_signs.systolic_bp and self.vital_signs.diastolic_bp:
                vitals.append(f"BP: {self.vital_signs.systolic_bp}/{self.vital_signs.diastolic_bp} mmHg")
            if self.vital_signs.heart_rate:
                vitals.append(f"HR: {self.vital_signs.heart_rate} bpm")
            if self.vital_signs.temperature:
                vitals.append(f"Temp: {self.vital_signs.temperature}°C")
            if self.vital_signs.oxygen_saturation:
                vitals.append(f"SpO2: {self.vital_signs.oxygen_saturation}%")
            if vitals:
                parts.append(f"\nVital Signs: {', '.join(vitals)}")
        
        if self.lab_results:
            lab_str = "\n".join([
                f"  - {lab.test_name}: {lab.value} {lab.unit or ''} ({lab.status or 'N/A'})"
                for lab in self.lab_results
            ])
            parts.append(f"\nLaboratory Results:\n{lab_str}")
        
        if self.imaging_results:
            parts.append(f"\nImaging Results: {', '.join(self.imaging_results)}")
        
        if self.physical_exam:
            parts.append(f"\nPhysical Examination:\n{self.physical_exam}")
        
        if self.free_text:
            parts.append(f"\nAdditional Notes:\n{self.free_text}")
        
        return "\n".join(parts)


class ClinicalRecommendation(BaseModel):
    """Structured clinical recommendation output"""
    assessment: str = Field(description="Clinical assessment or diagnosis")
    reasoning: str = Field(description="Clinical reasoning for the assessment")
    recommendations: List[str] = Field(description="Specific clinical recommendations")
    priority: str = Field(description="Priority level: urgent, high, moderate, low")
    follow_up: Optional[str] = Field(None, description="Recommended follow-up actions")
    confidence: Optional[str] = Field(None, description="Confidence level: high, moderate, low")
    
    def to_string(self) -> str:
        """Convert recommendation to formatted string"""
        parts = [
            f"Assessment: {self.assessment}",
            f"Reasoning: {self.reasoning}",
            f"Priority: {self.priority}",
            "\nRecommendations:",
        ]
        for i, rec in enumerate(self.recommendations, 1):
            parts.append(f"  {i}. {rec}")
        
        if self.follow_up:
            parts.append(f"\nFollow-up: {self.follow_up}")
        
        if self.confidence:
            parts.append(f"Confidence: {self.confidence}")
        
        return "\n".join(parts)

