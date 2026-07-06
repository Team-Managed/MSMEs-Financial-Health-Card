from backend.app.data.mock_generators import generate_profile
from backend.app.schemas.models import MSMEProfile

PERSONAS: dict[str, MSMEProfile] = {
    "healthy": generate_profile(
        seed=1001,
        sector="manufacturing",
        profile_type="healthy",
        msme_id="p001",
        business_name="Lakshmi Precision Parts",
        years_operating=8,
    ),
    "ntc": generate_profile(
        seed=1002,
        sector="services",
        profile_type="ntc",
        msme_id="p002",
        business_name="QuickServ Solutions",
        years_operating=2,
    ),
    "buyer_concentrated": generate_profile(
        seed=1003,
        sector="textiles",
        profile_type="buyer_concentrated",
        msme_id="p003",
        business_name="Weave & Craft Exports",
        years_operating=6,
    ),
    "seasonal": generate_profile(
        seed=1004,
        sector="agri-processing",
        profile_type="seasonal",
        msme_id="p004",
        business_name="Rabi Harvest Foods",
        years_operating=4,
    ),
}
