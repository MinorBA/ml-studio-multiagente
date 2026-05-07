# =============================================================================
# backend/matriz_confusion.py
# UTILIDAD: MatrizDeConfusion  (MC 2×2 manual)
# =============================================================================


class MatrizDeConfusion:
    """
    Calcula métricas de una MC 2×2:
    PG, EG, PP, PN, PFP, PFN, AP, AN.

    Parámetro
    ---------
    mc : list[list[int]] — [[VP, FP], [FN, VN]]
    """

    def __init__(self, mc: list):
        self._mc = mc

    @property
    def mc(self) -> list:
        return self._mc

    @mc.setter
    def mc(self, v: list):
        self._mc = v

    def calcular(self) -> dict:
        mc = self._mc
        if len(mc) != 2 or len(mc[0]) != 2:
            return {"error": "La matriz no es 2×2"}
        VP, FP = mc[0][0], mc[0][1]
        FN, VN = mc[1][0], mc[1][1]
        total  = VP + VN + FP + FN
        pg  = (VP + VN) / total
        eg  = 1 - pg
        pp  = VP / (VP + FP) if (VP + FP) > 0 else 0
        pn  = VN / (VN + FN) if (VN + FN) > 0 else 0
        pfp = FP / (VN + FP) if (VN + FP) > 0 else 0
        pfn = FN / (VP + FN) if (VP + FN) > 0 else 0
        ap  = VP / (VP + FN) if (VP + FN) > 0 else 0
        an  = VN / (VN + FP) if (VN + FP) > 0 else 0
        return {
            "Precisión Global":   pg,
            "Error Global":       eg,
            "Precisión Positiva": pp,
            "Precisión Negativa": pn,
            "Falsos Positivos":   pfp,
            "Falsos Negativos":   pfn,
            "Asertividad Positiva": ap,
            "Asertividad Negativa": an,
        }
