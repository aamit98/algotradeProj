# Portfolio Studio

אפליקציה אינטראקטיבית לבניית פורטפוליו מותאם אישית והערכתו באמצעות סימולציית Monte Carlo ו־ARIMA.

---

## 📖 תיאור הפרויקט
''
- Web‐app מבוסס **Streamlit**  
- סימולציית **Monte Carlo** ליצירת תמהילי משקלים אקראיים  
- חיזוי סדרות זמן לטווח של 90 ימים בעזרת **ARIMA** עם מנגנון fallback  
- הצגת **חזית יעילה** (Efficient Frontier) וגרף תחזיות עם מרווחי ביטחון  
- העלאת קובץ CSV קיים (`ticker,weight`) והשוואה גרפית  
- ממשק ידידותי עם **CSS מותאם** ו-**Session State** להצגת מידע דינמי  

---

## 🚀 תכונות עיקריות

1. **שאלון סיכון** – אבחון אוטומטי של פרופיל הסיכון  
2. **Monte Carlo Simulation** – אלפי פורטפוליו בצורה וקטורית  
3. **ARIMA Forecasting** – תחזית יומית ל-90 ימים + fallback פשוט  
4. **Efficient Frontier** – פיזור נקודות תשואה/סיכון + סימון הפורטפוליו המומלץ  
5. **CSV Compare** – השוואה בין תמהיל מומלץ לתמהיל קיים  
6. **Learn Tickers** – כפתור להצגת מידע על ניירות ערך נבחרים  
7. **Caching** – טעינת מחירי שוק מהירה עם `@st.cache_data`  
8. **Responsive Charts** – גרפים מותאמים למובייל עם `use_container_width=True` ו-DateLocator  

---

## 🛠️ דרישות מערכת

- Python 3.8+  
- Streamlit  
- pandas, NumPy, matplotlib  
- yfinance  
- statsmodels (ARIMA)  

---

## ⚙️ התקנה והרצה

```bash
# 1. קלון הריפוזיטורי
git clone https://github.com/aamit98/algotradeProj.git
cd algotradeProj

# 2. יצירת Virtual Environment
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Linux/macOS:
source .venv/bin/activate

# 3. התקנת תלויות
pip install -r requirements.txt

# 4. הרצת האפליקציה
streamlit run Skeleton.py

# 5. פתח בדפדפן
http://localhost:8501
📂 מבנה התיקיות
bash
Copy
Edit
/
├── Skeleton.py               # קוד המקור הראשי של Streamlit
├── requirements.txt     # רשימת חבילות Python
├── readme.md            # קובץ זה

