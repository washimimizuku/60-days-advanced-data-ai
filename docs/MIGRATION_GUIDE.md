# Curriculum Migration Guide - Optimized Structure

## What Changed?

The 60 Days Advanced curriculum has been **restructured** to move orchestration earlier and improve learning progression.

### Major Changes

1. **Orchestration moved from Days 51-57 to Days 12-21** ⭐
2. **Feature stores moved from Day 12 to Day 25** (with ML)
3. **AWS data services moved from Days 58-59 to Days 22-23**
4. **All day numbers after Day 14 have shifted**

---

## Day Number Mapping

### Phase 1: Production Data Engineering
| Old Day | New Day | Topic | Change |
|---------|---------|-------|--------|
| 1-11 | 1-11 | Same | ✅ No change |
| 12 | 25 | Feature stores | ⚠️ Moved to ML section |
| 13 | - | Apache Flink | ❌ Removed |
| - | 12 | Airflow basics | ⭐ NEW |
| - | 13 | dbt basics | ⭐ NEW |
| 14 | 14 | Project | ✅ Enhanced with Airflow |

### Phase 2: Data Orchestration & Quality (NEW)
| Old Day | New Day | Topic | Change |
|---------|---------|-------|--------|
| 51 | 15 | Airflow production patterns | ⚠️ Moved up 36 days |
| 52 | 16 | Airflow at scale | ⚠️ Moved up 36 days |
| 53 | 17 | dbt deep dive | ⚠️ Moved up 36 days |
| 54 | 18 | dbt advanced | ⚠️ Moved up 36 days |
| 55 | 19 | Data quality | ⚠️ Moved up 36 days |
| 56 | 20 | Data observability | ⚠️ Moved up 36 days |
| - | 21 | Testing strategies | ⭐ NEW |
| 58 | 22 | AWS Glue | ⚠️ Moved up 36 days |
| 59 | 23 | AWS Kinesis | ⚠️ Moved up 36 days |
| 57 | 24 | Project | ⚠️ Moved up 33 days |

### Phase 3: Advanced ML & MLOps
| Old Day | New Day | Topic | Change |
|---------|---------|-------|--------|
| - | 25 | Feature stores | ⚠️ Moved from Day 12 |
| 16 | 26 | Advanced feature engineering | ⚠️ +10 days |
| 17 | 27 | Time series | ⚠️ +10 days |
| 18 | 28 | Anomaly detection | ⚠️ +10 days |
| 19 | 29 | Recommendation systems | ⚠️ +10 days |
| 20 | 30 | Ensemble methods | ⚠️ +10 days |
| 21 | 31 | Model explainability | ⚠️ +10 days |
| 22 | 32 | Project | ⚠️ +10 days |
| 23 | 33 | Model serving | ⚠️ +10 days |
| 24 | 34 | A/B testing | ⚠️ +10 days |
| 25 | 35 | Model versioning | ⚠️ +10 days |
| 26 | 36 | CI/CD for ML | ⚠️ +10 days |
| 27 | 37 | Feature monitoring | ⚠️ +10 days |
| 28 | 38 | AutoML | ⚠️ +10 days |
| 29 | 39 | Project | ⚠️ +10 days |
| 30 | 40 | Checkpoint | ⚠️ +10 days |

### Phase 4: Advanced GenAI & LLMs
| Old Day | New Day | Topic | Change |
|---------|---------|-------|--------|
| 31 | 41 | Transformer architecture | ⚠️ +10 days |
| 32 | 42 | Attention mechanisms | ⚠️ +10 days |
| 33 | 43 | Tokenization | ⚠️ +10 days |
| 34 | 44 | LLM training | ⚠️ +10 days |
| 35 | 45 | Prompt engineering | ⚠️ +10 days |
| 36 | 46 | Prompt security | ⚠️ +10 days |
| 37 | 47 | Project | ⚠️ +10 days |
| 38 | 48 | Fine-tuning | ⚠️ +10 days |
| 39 | 49 | RLHF and DPO | ⚠️ +10 days |
| 40 | 50 | Quantization | ⚠️ +10 days |
| 41 | 51 | LLM serving | ⚠️ +10 days |
| 42 | 52 | Advanced RAG | ⚠️ +10 days |
| 43 | 53 | RAG evaluation | ⚠️ +10 days |
| 44 | 54 | Project | ⚠️ +10 days |

### Phase 5: Infrastructure & DevOps
| Old Day | New Day | Topic | Change |
|---------|---------|-------|--------|
| 45 | 55 | AWS deep dive | ⚠️ +10 days |
| 46 | 56 | Kubernetes | ⚠️ +10 days |
| 47 | 57 | Terraform | ⚠️ +10 days |
| 48 | 58 | Monitoring | ⚠️ +10 days |
| 49 | 59 | Cost optimization | ⚠️ +10 days |
| 50 | - | Checkpoint | ❌ Removed |
| 60 | 60 | Capstone | ✅ Same |

---

## For Current Users

### If You're on Day 1-11
✅ **No changes** - Continue as planned

### If You're on Day 12-14
⚠️ **Changes ahead**:
- Old Day 12 (Feature stores) → Skip for now, will do on Day 25
- Old Day 13 (Flink) → Skip (removed from curriculum)
- **NEW Day 12**: Airflow basics
- **NEW Day 13**: dbt basics
- Day 14: Project (now includes Airflow)

**Action**: 
1. Skip old Days 12-13
2. Do NEW Days 12-13 (Airflow + dbt basics)
3. Continue with Day 14

### If You're on Day 15-50
⚠️ **Major shift**:
- All days shifted by +10
- Old Day 16 → New Day 26
- Old Day 30 → New Day 40
- Old Day 44 → New Day 54

**Action**:
1. Note your current day number (e.g., Day 20)
2. Add 10 to get new day number (Day 30)
3. Continue from new day number
4. **Before continuing**: Do Days 12-24 (orchestration section)

### If You're on Day 51-60
⚠️ **Content moved**:
- Days 51-57 (Orchestration) → Now Days 15-21
- Days 58-59 (AWS Glue/Kinesis) → Now Days 22-23
- Day 60 (Capstone) → Still Day 60

**Action**:
1. You've already done orchestration content
2. Skip Days 15-23 (already covered)
3. Continue with infrastructure (Days 55-60)

---

## Migration Paths

### Path 1: Start Fresh (Recommended for Days 1-14)
If you're in the first 2 weeks, **restart with new curriculum**:
- Better learning progression
- Orchestration from the start
- All projects use proper tools

### Path 2: Skip Ahead (For Days 15-50)
If you're in the middle:
1. **Pause** at your current day
2. **Go back** and do Days 12-24 (orchestration)
3. **Resume** at your day + 10

Example: Currently on Day 25 (Model versioning)
1. Pause
2. Do Days 12-24 (orchestration - 13 days)
3. Resume at Day 35 (Model versioning in new curriculum)

### Path 3: Continue and Backfill (For Days 51-60)
If you're near the end:
1. **Continue** with infrastructure (Days 55-60)
2. **Skip** Days 15-23 (you've already done this content)
3. **Complete** capstone with orchestration knowledge

---

## Why This Change?

### Problem with Old Structure
- ❌ Learned orchestration AFTER building most projects
- ❌ Had to retrofit orchestration into existing projects
- ❌ Not realistic for production work
- ❌ Feature stores before ML fundamentals

### Benefits of New Structure
- ✅ Orchestration available from Day 14
- ✅ All projects after Day 14 use Airflow/dbt
- ✅ Feature stores introduced with ML context
- ✅ AWS data services grouped logically
- ✅ More production-ready from earlier
- ✅ Better learning progression

---

## Quick Reference: What to Do

| Your Current Day | Action |
|-----------------|--------|
| **Days 1-11** | ✅ Continue, no changes |
| **Day 12** | ⚠️ Skip old Day 12, do NEW Day 12 (Airflow) |
| **Day 13** | ⚠️ Skip old Day 13, do NEW Day 13 (dbt) |
| **Day 14** | ✅ Continue (enhanced with Airflow) |
| **Days 15-50** | ⚠️ Pause, do Days 12-24, resume at +10 |
| **Days 51-57** | ⚠️ Content moved to Days 15-21 |
| **Days 58-59** | ⚠️ Content moved to Days 22-23 |
| **Day 60** | ✅ Continue (same) |

---

## FAQ

**Q: Do I have to restart?**
A: No, but recommended if you're in Days 1-14. Otherwise, follow migration path.

**Q: What if I already did Days 51-57?**
A: Skip Days 15-21 in new curriculum, you've already covered it.

**Q: Why remove Apache Flink?**
A: Less commonly used than Airflow/dbt. Focus on most valuable skills.

**Q: Will projects change?**
A: Projects enhanced to use orchestration, but core concepts same.

**Q: How long to migrate?**
A: If following Path 2, add 13 days (Days 12-24) to your timeline.

---

## Support

If you have questions about migration:
1. Check the new CURRICULUM.md
2. Review CURRICULUM_OPTIMIZATION_ANALYSIS.md for rationale
3. Follow the migration path for your current day

The new structure provides a much better learning experience! 🎯
