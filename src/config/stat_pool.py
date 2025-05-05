class StatDefinition:
    def __init__(self, name, min_value, max_value, type="fixe", weight=1):
        self.name = name
        self.min_value = min_value
        self.max_value = max_value
        self.type = type
        self.weight = weight

    def __repr__(self):
        return f"{self.name} ({self.min_value}-{self.max_value}, poids={self.weight})"

stat_pool = {
    "vitalité": StatDefinition("vitalité", 0, 505, weight=0.2),
    "sagesse": StatDefinition("sagesse", 0, 60, weight=3),
    "force": StatDefinition("force", 0, 101, weight=1),
    "chance": StatDefinition("chance", 0, 101, weight=1),
    "agilité": StatDefinition("agilité", 0, 101, weight=1),
    "intelligence": StatDefinition("intelligence", 0, 101, weight=1),
    "dommage": StatDefinition("dommages", 0, 20, weight=20),
    "dommage_neutre" : StatDefinition("dommage_neutre", 0, 20, weight = 5),
    "dommage_terre" : StatDefinition("dommage_terre", 0, 20, weight = 5),
    "dommage_feu" : StatDefinition("dommage_feu", 0, 20, weight = 5),
    "dommage_eau" : StatDefinition("dommage_eau", 0, 20, weight = 5),
    "dommage_air" : StatDefinition("dommage_air", 0, 20, weight = 5),
    "dommage_critique" : StatDefinition("dommage_critique", 0, 25, weight = 5),
    "puissance": StatDefinition("puissance", 0, 100, weight=2),
    "coups_critiques": StatDefinition("coups_critiques", 0, 10, weight=10),
    "invocation": StatDefinition("invocation", 0, 2, weight=30),
    "résistance_neutre": StatDefinition("résistance_neutre", 0, 25, weight=2),
    "résistance_terre": StatDefinition("résistance_terre", 0, 25, weight=2),
    "résistance_feu": StatDefinition("résistance_feu", 0, 25, weight=2),
    "résistance_eau": StatDefinition("résistance_eau", 0, 25, weight=2),
    "résistance_air": StatDefinition("résistance_air", 0, 25, weight=2),
    "résistance_poussée": StatDefinition("résistance_poussée", 0, 80, weight=2),
    "résistance_critique": StatDefinition("résistance_critique", 0, 40, weight=2),
    "résistance_neutre_%": StatDefinition("résistance_neutre_%", 0, 15, weight=6),
    "résistance_terre_%": StatDefinition("résistance_terre_%", 0, 15, weight=6),
    "résistance_feu_%": StatDefinition("résistance_feu_%", 0, 15, weight=6),
    "résistance_eau_%": StatDefinition("résistance_eau_%", 0, 15, weight=6),
    "résistance_air_%": StatDefinition("résistance_air_%", 0, 15, weight=6),
    "tacle": StatDefinition("tacle", 0, 20, weight=4),
    "fuite": StatDefinition("fuite", 0, 20, weight=4),
    "soin": StatDefinition("soins", 0, 30, weight=10),
    "PA": StatDefinition("PA", 0, 1, weight=100),
    "PM": StatDefinition("PM", 0, 1, weight=90),
    "PO": StatDefinition("PO", 0, 2, weight=51),
    "prospection": StatDefinition("prospection", 0, 33, weight=3),
    "initiative": StatDefinition("initiative", 0, 800, weight=0.1),
    "pods": StatDefinition("pods", 0, 2000, weight=0.25),
    "dommage_dist_%" : StatDefinition("dommage_dist_%", 0, 5, weight=15),
    "dommage_sort_%" : StatDefinition("dommage_sort_%", 0, 5, weight=15),
    "dommage_mêlee_%" : StatDefinition("dommage_mêlee_%", 0, 5, weight=15),
    "résistance_dist_%" : StatDefinition("résistance_dist_%", 0, 5, weight=15),
    "résistance_sort_%" : StatDefinition("résistance_sort_%", 0, 5, weight=15),
    "résistance_melee_%" : StatDefinition("résistance_melee_%", 0, 5, weight=15)
}

essential_stats = {
    "PA", "PM", "PO", "invocation",
    "coups_critiques",
    "résistance_dist_%", "résistance_sort_%", "résistance_mêlée_%", 
    "dommage_mêlée_%", "dommage_dist_%", "dommage_sort_%"
}

basic_stats = {
    "vitalité", "force", "chance", "agilité", "intelligence",
    "puissance"
}

secondary_stats = {
    "sagesse", "tacle", "fuite", "prospection",
    "initiative", "pods", "soin"
}

heavy_stats = {
    "PA", "PM", "PO", "invocation",
    "sagesse", "coup_critiques", 
    "résistance_dist_%", "résistance_sort_%", "résistance_mêlée_%", 
    "dommage_mêlée_%", "dommage_dist_%", "dommage_sort_%"
}

résistance_pourcent_stats = {
    "résistance_neutre_%", "résistance_terre_%", "résistance_feu_%", "résistance_eau_%", "résistance_air_%",
    "résistance_dist_%", "résistance_sort_%", "résistance_mêlée_%", 
}

résistance_elem_stats = {
    "résistance_neutre", "résistance_terre", "résistance_feu", "résistance_eau", "résistance_air", 
    "résistance_poussée", "résistance_critique"
}

dommage_elem_stats = {
    "dommage_neutre", "dommage_terre", "dommage_feu", "dommage_eau", "dommage_air", 
    "dommage_critique", "dommage_poussée"
}

dommage_pourcent_stats = {
    "dommage_mêlée_%", "dommage_dist_%", "dommage_sort_%"
}