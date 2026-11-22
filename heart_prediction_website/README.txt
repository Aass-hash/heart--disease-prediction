# Heart Disease Prediction System

## تشغيل الموقع على جهازك

1. ثبّت Python.
2. افتح Terminal أو Command Prompt داخل هذا المجلد.
3. نفّذ:
   `pip install -r requirements.txt`
4. شغّل:
   `streamlit run app.py`
5. سيفتح الموقع في المتصفح تلقائياً.

## الملفات
- `app.py`: واجهة الموقع.
- `ann_model.joblib`: نموذج ANN المدرّب من بيانات المشروع.
- `requirements.txt`: المكتبات المطلوبة.

## ملاحظة مهمة
تم اتباع ترميز الهدف الموجود في ملف Colab:
`<50 = 0` و `>50_1 = 1`.
هذا نموذج تعليمي وليس أداة تشخيص طبي.
