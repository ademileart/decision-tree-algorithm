from enum import Enum


class Gender(Enum):
    MALE = "Male"
    FEMALE = "Female"
    OTHER = "Other"


class Hypertension(Enum):
    NO_HYPERTENSION = 0
    HYPERTENSION = 1


class HeartDisease(Enum):
    NO_HEART_DISEASE = 0
    HEART_DISEASE = 1


class Married(Enum):
    NO = 0
    YES = 1


class WorkType(Enum):
    CHILDREN = "Children"
    GOVERNMENT_JOB = "Govt_job"
    NEVER_WORKED = "Never_worked"
    PRIVATE = "Private"
    SELF_EMPLOYED = "Self_employed"


class ResidenceType(Enum):
    URBAN = "Urban"
    RURAL = "Rural"
