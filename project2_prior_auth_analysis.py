# ============================================================
# PROJECT 2: Prior Auth Approval Rate Analysis
# Tools: SQLite (SQL) + Python (Pandas, SciPy, Sklearn)
# Author: Rohit Bhandari
# ============================================================

import sqlite3, pandas as pd, numpy as np
import matplotlib.pyplot as plt, seaborn as sns
from datetime import datetime, timedelta
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import random, warnings
warnings.filterwarnings('ignore')

DB_PATH = '/home/claude/rohit_projects/prior_auth.db'

# ── STEP 1: Create Database Schema ──────────────────────────
print("=" * 60)
print("STEP 1: Creating SQLite Database & Schema")
print("=" * 60)

conn = sqlite3.connect(DB_PATH)
cur  = conn.cursor()
cur.executescript('''
DROP TABLE IF EXISTS prior_auth;
CREATE TABLE prior_auth (
    auth_id          TEXT PRIMARY KEY,
    submission_date  TEXT,
    insurer          TEXT,
    procedure_code   TEXT,
    department       TEXT,
    status           TEXT,
    processing_days  INTEGER,
    denial_reason    TEXT,
    resubmitted      INTEGER,
    final_approved   INTEGER
);
''')
conn.commit()
print("  -> Schema created: prior_auth table")

# ── STEP 2: Populate with 2,400 Realistic Records ───────────
print("\n" + "=" * 60)
print("STEP 2: Inserting 2,400 Records")
print("=" * 60)

np.random.seed(99)
n = 2400
insurers    = ['UnitedHealth', 'Aetna', 'BlueCross', 'Cigna', 'Humana']
proc_codes  = ['99213', '99214', '45378', '70553', '93000', '27447']
depts       = ['Oncology', 'Gastroenterology', 'Cardiology', 'Radiology', 'Orthopedics']
deny_rsns   = ['Missing Documentation', 'Not Medically Necessary',
               'Out of Network', 'Duplicate Request', None]
# Approval rates differ by insurer to create meaningful patterns
insurer_probs = {
    'UnitedHealth': [0.72, 0.15, 0.07, 0.06],
    'Aetna':        [0.65, 0.22, 0.07, 0.06],
    'BlueCross':    [0.75, 0.13, 0.07, 0.05],
    'Cigna':        [0.60, 0.25, 0.08, 0.07],
    'Humana':       [0.70, 0.17, 0.08, 0.05],
}

rows = []
for i in range(1, n+1):
    insurer  = random.choice(insurers)
    probs    = insurer_probs[insurer]
    status   = np.random.choice(['Approved','Denied','Pending','Appealed'], p=probs)
    denied   = 1 if status == 'Denied'  else 0
    resubmit = 1 if denied and random.random() > 0.4 else 0
    approved = 1 if (status == 'Approved') or (resubmit and random.random() > 0.3) else 0
    date_str = (datetime(2024,1,1)+timedelta(days=random.randint(0,364))).strftime('%Y-%m-%d')
    rows.append((
        f'PA{i:05d}', date_str, insurer,
        random.choice(proc_codes), random.choice(depts), status,
        random.randint(1, 30),
        random.choice(deny_rsns) if denied else None,
        resubmit, approved
    ))

cur.executemany('INSERT INTO prior_auth VALUES (?,?,?,?,?,?,?,?,?,?)', rows)
conn.commit()
print(f"  -> Inserted {n} records into prior_auth")

# ── STEP 3: SQL Analysis ─────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: SQL KPI Queries")
print("=" * 60)

queries = {
    "Overall Approval Rate": '''
        SELECT
            COUNT(*) AS total_submissions,
            SUM(final_approved) AS total_approved,
            ROUND(100.0 * SUM(final_approved) / COUNT(*), 2) AS approval_pct
        FROM prior_auth''',

    "Approval Rate by Insurer": '''
        SELECT insurer,
               COUNT(*) AS submissions,
               SUM(final_approved) AS approved,
               ROUND(100.0*SUM(final_approved)/COUNT(*), 2) AS approval_pct,
               ROUND(AVG(processing_days), 1) AS avg_processing_days
        FROM prior_auth
        GROUP BY insurer
        ORDER BY approval_pct DESC''',

    "Top Denial Reasons": '''
        SELECT denial_reason,
               COUNT(*) AS count,
               ROUND(100.0*COUNT(*)/
                   (SELECT COUNT(*) FROM prior_auth WHERE status='Denied'), 2) AS pct_of_denials
        FROM prior_auth
        WHERE status = 'Denied' AND denial_reason IS NOT NULL
        GROUP BY denial_reason
        ORDER BY count DESC''',

    "Monthly Submission Trend": '''
        SELECT substr(submission_date,1,7) AS month,
               COUNT(*) AS submissions,
               ROUND(100.0*SUM(final_approved)/COUNT(*), 2) AS approval_pct
        FROM prior_auth
        GROUP BY month
        ORDER BY month''',
}

results = {}
for title, sql in queries.items():
    df_q = pd.read_sql(sql, conn)
    results[title] = df_q
    print(f"\n  [{title}]")
    print(df_q.to_string(index=False))

# ── STEP 4: Advanced Window Function ────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Advanced SQL - Window Functions")
print("=" * 60)

dept_rank_sql = '''
    SELECT department, insurer,
           COUNT(*) AS submissions,
           ROUND(100.0*SUM(final_approved)/COUNT(*),2) AS approval_pct,
           RANK() OVER (PARTITION BY department
                        ORDER BY 100.0*SUM(final_approved)/COUNT(*) DESC) AS dept_rank
    FROM prior_auth
    GROUP BY department, insurer
    ORDER BY department, dept_rank
'''
dept_rank = pd.read_sql(dept_rank_sql, conn)
print(dept_rank.to_string(index=False))

# ── STEP 5: Statistical Tests ────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Statistical Analysis")
print("=" * 60)

df = pd.read_sql('SELECT * FROM prior_auth', conn)
df['submission_date'] = pd.to_datetime(df['submission_date'])
df['month'] = df['submission_date'].dt.to_period('M')

# Chi-square: approval rates differ across insurers?
contingency = pd.crosstab(df['insurer'], df['final_approved'])
chi2, p, dof, _ = stats.chi2_contingency(contingency)
print(f"  Chi-square test (approval rate by insurer):")
print(f"    Chi2 = {chi2:.2f}, df = {dof}, p = {p:.6f}")
print(f"    Conclusion: {'Significant difference (p<0.05)' if p<0.05 else 'No significant difference'}")

# Kruskal-Wallis: processing days differ across insurers?
groups = [df[df['insurer']==ins]['processing_days'].values for ins in df['insurer'].unique()]
h_stat, kw_p = stats.kruskal(*groups)
print(f"\n  Kruskal-Wallis test (processing days by insurer):")
print(f"    H = {h_stat:.2f}, p = {kw_p:.4f}")

# Mann-Whitney: approved vs denied processing time
approved_days = df[df['final_approved']==1]['processing_days']
denied_days   = df[df['final_approved']==0]['processing_days']
u_stat, mw_p = stats.mannwhitneyu(approved_days, denied_days, alternative='two-sided')
print(f"\n  Mann-Whitney (processing days: approved vs denied):")
print(f"    Approved median = {approved_days.median():.1f} days")
print(f"    Denied   median = {denied_days.median():.1f} days")
print(f"    U = {u_stat:.0f}, p = {mw_p:.4f}")

# ── STEP 6: Machine Learning - Denial Risk Model ─────────────
print("\n" + "=" * 60)
print("STEP 6: Predictive Model - Denial Risk Score")
print("=" * 60)

df_model = df.copy()
for col in ['insurer', 'procedure_code', 'department']:
    df_model[col] = LabelEncoder().fit_transform(df_model[col])

features = ['insurer', 'procedure_code', 'department', 'processing_days']
X = df_model[features]
y = df_model['final_approved']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]
auc    = roc_auc_score(y_test, y_prob)

print(classification_report(y_test, y_pred, target_names=['Denied','Approved']))
print(f"  ROC-AUC Score: {auc:.3f}")

feat_imp = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
print(f"\n  Feature Importances:")
for feat, imp in feat_imp.items():
    print(f"    {feat:<22} {imp:.3f}")

# ── STEP 7: Visualisations ───────────────────────────────────
print("\n" + "=" * 60)
print("STEP 7: Generating Analysis Charts")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle('Prior Authorisation Approval Rate Analysis - 2024\nRohit Bhandari | Healthcare Data Analyst',
             fontsize=14, fontweight='bold', y=0.99)

# Plot 1: Approval Rate by Insurer
ins_stats = df.groupby('insurer')['final_approved'].mean().mul(100).sort_values()
colors1 = ['#c00000' if v < 70 else '#FF7B00' if v < 75 else '#70AD47' for v in ins_stats.values]
axes[0,0].barh(ins_stats.index, ins_stats.values, color=colors1, edgecolor='white')
axes[0,0].axvline(95, color='red', linestyle='--', linewidth=1.5, label='Target 95%')
axes[0,0].axvline(ins_stats.mean(), color='blue', linestyle=':', linewidth=1.2, label=f'Average {ins_stats.mean():.1f}%')
for i, v in enumerate(ins_stats.values):
    axes[0,0].text(v+0.3, i, f'{v:.1f}%', va='center', fontsize=9)
axes[0,0].set_title('Approval Rate by Insurer (%)', fontweight='bold')
axes[0,0].set_xlabel('Approval Rate (%)'); axes[0,0].legend(fontsize=8)

# Plot 2: Monthly Trend
monthly = df.groupby('month')['final_approved'].mean().mul(100)
axes[0,1].plot(range(len(monthly)), monthly.values, color='#1F4E79',
               marker='o', linewidth=2, markersize=5)
axes[0,1].axhline(95, color='red', linestyle='--', linewidth=1.2, label='Target 95%')
axes[0,1].fill_between(range(len(monthly)), monthly.values, 95,
                        where=[v<95 for v in monthly.values], alpha=0.2, color='red', label='Below target')
axes[0,1].set_title('Monthly Approval Rate Trend (%)', fontweight='bold')
axes[0,1].set_ylabel('Approval Rate (%)'); axes[0,1].legend(fontsize=8)
axes[0,1].set_xticks(range(0,len(monthly),2))
axes[0,1].set_xticklabels([str(m) for m in monthly.index[::2]], rotation=45, fontsize=7)

# Plot 3: Denial Reasons Pie
deny = df[df['status']=='Denied']['denial_reason'].dropna().value_counts()
wedge_props = dict(width=0.5, edgecolor='white')
axes[0,2].pie(deny.values, labels=deny.index, autopct='%1.1f%%',
              startangle=90, wedgeprops=wedge_props,
              colors=['#1F4E79','#2E75B6','#9DC3E6','#BDD7EE'])
axes[0,2].set_title('Denial Reason Breakdown', fontweight='bold')

# Plot 4: Processing Days: Approved vs Denied
sns.histplot(data=df, x='processing_days', hue='final_approved',
             bins=20, ax=axes[1,0], palette={0:'#c00000', 1:'#70AD47'}, alpha=0.7)
axes[1,0].set_title('Processing Days: Approved vs Denied', fontweight='bold')
axes[1,0].set_xlabel('Processing Days')
legend = axes[1,0].get_legend()
if legend: legend.set_title(''); [t.set_text(l) for t,l in zip(legend.texts, ['Denied','Approved'])]

# Plot 5: Approval Rate by Department x Insurer (heatmap)
pivot = df.groupby(['department','insurer'])['final_approved'].mean().mul(100).unstack()
sns.heatmap(pivot, ax=axes[1,1], annot=True, fmt='.0f', cmap='RdYlGn',
            vmin=55, vmax=85, linewidths=0.4, cbar_kws={'shrink':0.7})
axes[1,1].set_title('Approval Rate (%) by Dept x Insurer', fontweight='bold')
axes[1,1].set_xlabel('Insurer'); axes[1,1].set_ylabel('Department')

# Plot 6: Feature Importance + ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
axes[1,2].plot(fpr, tpr, color='#1F4E79', linewidth=2, label=f'ROC (AUC={auc:.2f})')
axes[1,2].plot([0,1],[0,1], 'k--', linewidth=1)
axes[1,2].set_title('Model ROC Curve (Denial Risk Model)', fontweight='bold')
axes[1,2].set_xlabel('False Positive Rate'); axes[1,2].set_ylabel('True Positive Rate')
axes[1,2].legend(fontsize=9)

plt.tight_layout()
plt.savefig('/home/claude/rohit_projects/prior_auth_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("  -> Saved: prior_auth_analysis.png")

# ── STEP 8: Export Findings Report ──────────────────────────
print("\n" + "=" * 60)
print("STEP 8: Exporting Findings to Excel")
print("=" * 60)

with pd.ExcelWriter('/home/claude/rohit_projects/prior_auth_report.xlsx', engine='openpyxl') as writer:
    results["Approval Rate by Insurer"].to_excel(writer, sheet_name='By Insurer',    index=False)
    results["Top Denial Reasons"].to_excel(      writer, sheet_name='Denial Reasons',index=False)
    results["Monthly Submission Trend"].to_excel(writer, sheet_name='Monthly Trend', index=False)
    dept_rank.to_excel(                          writer, sheet_name='Dept Rankings',  index=False)
    df[['auth_id','submission_date','insurer','department',
        'status','processing_days','final_approved']].to_excel(
        writer, sheet_name='Raw Data', index=False)
print("  -> Saved: prior_auth_report.xlsx")

print("\nPROJECT 2 COMPLETE")
print("Files generated:")
print("  1. prior_auth.db           (SQLite database)")
print("  2. prior_auth_analysis.png (6-chart analysis)")
print("  3. prior_auth_report.xlsx  (Excel findings)")
