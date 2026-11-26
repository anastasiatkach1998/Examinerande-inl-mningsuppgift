import numpy as np
from scipy.stats import ttest_ind
from sklearn.linear_model import LinearRegression

def statistik(df):
    """Tar fram beskrivande statistik för valda kolumner."""
    kolumner = ["age", "weight", "height", "systolic_bp", "cholesterol"]
    data = df[kolumner].describe()
    return data

def sjukdomsandel(df):
    """Beräknar andel personer med sjukdom."""
    andel_sjukdom = df["disease"].mean()
    return andel_sjukdom

def konfidensintervall(data):
    """Beräknar 95% konfidentintervall med normalapproximation."""
    mean = data.mean()
    std = data.std()
    se = std / np.sqrt(len(data))
    felmarginal = se * 1.96
    nedre_grans = mean - felmarginal
    ovre_grans = mean + felmarginal
    return mean, nedre_grans, ovre_grans 

def hypotesprovning(grupp1, grupp2):
    """Utför ensidigt Welch t-test mellan två grupper."""
    t_stat, p_value = ttest_ind(
    grupp1, grupp2,
    equal_var=False,
    alternative="greater"    
)
    return t_stat, p_value


class HealthAnalyzer:
    """Utför statistiska analyser av hälsodata."""

    def __init__(self, df): 
        """Sparar datasetet i objektet."""
        self.df = df

    def statistik(self):
        """Tar fram beskrivande statistik."""
        return statistik(self.df)

    def konfidensintervall(self,kolumn):
        """Beräknar konfidensintervall för valda kolumn."""
        return konfidensintervall(self.df[kolumn])

    def enkel_regression(self):
        """Samband mellan ålder och blodtryck."""
        X = self.df[["age"]]
        y = self.df["systolic_bp"]
        modell = LinearRegression()
        modell.fit(X, y)
        skarning = float(modell.intercept_)
        lutning = float(modell.coef_[0])
        y_pred = modell.predict(X)
        r2 = modell.score(X, y)
        return y_pred, r2, skarning, lutning 

    def multipel_regression(self):
        """Förutsäger blodtryck utifrån ålder, vikt och kolesterol"""
        X = self.df[["age", "weight", "cholesterol"]]
        y = self.df["systolic_bp"]
        modell = LinearRegression()
        modell.fit(X, y)
        y_pred = modell.predict(X)
        r2 = modell.score(X, y)
        return y_pred, r2
