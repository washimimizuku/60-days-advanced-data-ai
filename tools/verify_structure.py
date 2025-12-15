#!/usr/bin/env python3
"""
Verify 60-day curriculum structure is complete.
"""

import os
from pathlib import Path

def verify_structure():
    """Verify all days have required files"""
    
    days_dir = Path('days')
    required_files = ['README.md', 'quiz.md']
    optional_files = ['exercise.py', 'solution.py', 'project.md', 'exercise.sql', 'solution.sql', 'review.md']
    
    issues = []
    success = []
    
    for day_num in range(1, 61):
        day_folder = days_dir / f'day-{day_num:02d}'
        
        if not day_folder.exists():
            issues.append(f"❌ Day {day_num}: Folder missing")
            continue
        
        # Check required files
        has_readme = (day_folder / 'README.md').exists()
        has_quiz = (day_folder / 'quiz.md').exists()
        
        # Check for at least one exercise/solution file
        has_exercise = any((day_folder / f).exists() for f in optional_files)
        
        if not has_readme:
            issues.append(f"❌ Day {day_num}: Missing README.md")
        elif not has_quiz:
            issues.append(f"⚠️  Day {day_num}: Missing quiz.md")
        elif not has_exercise:
            issues.append(f"⚠️  Day {day_num}: Missing exercise/solution files")
        else:
            success.append(f"✅ Day {day_num}: Complete")
    
    # Print results
    print("=" * 60)
    print("60-DAY CURRICULUM STRUCTURE VERIFICATION")
    print("=" * 60)
    print()
    
    if success:
        print(f"✅ COMPLETE: {len(success)}/60 days")
        print()
    
    if issues:
        print(f"⚠️  ISSUES FOUND: {len(issues)}")
        print()
        for issue in issues:
            print(issue)
        print()
    else:
        print("🎉 ALL 60 DAYS ARE COMPLETE!")
        print()
    
    # Check key restructured days
    print("=" * 60)
    print("KEY RESTRUCTURED DAYS")
    print("=" * 60)
    print()
    
    key_days = {
        12: "Airflow basics (NEW)",
        13: "dbt basics (NEW)",
        15: "Airflow production patterns (NEW)",
        24: "Project - Airflow + dbt + quality (NEW)",
        25: "Feature stores (MOVED from Day 12)",
    }
    
    for day_num, description in key_days.items():
        day_folder = days_dir / f'day-{day_num:02d}'
        if day_folder.exists() and (day_folder / 'README.md').exists():
            print(f"✅ Day {day_num}: {description}")
        else:
            print(f"❌ Day {day_num}: {description} - MISSING")
    
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total days: 60")
    print(f"Complete: {len(success)}")
    print(f"Issues: {len(issues)}")
    print(f"Success rate: {len(success)/60*100:.1f}%")
    print()
    
    if len(success) == 60:
        print("🎉 RESTRUCTURE COMPLETE AND VERIFIED!")
        return True
    else:
        print("⚠️  Some issues need attention")
        return False

if __name__ == "__main__":
    verify_structure()
