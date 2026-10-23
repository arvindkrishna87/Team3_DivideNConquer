# Stage 3 split validation

Seed: 3902026
Eligible cases: 93
All checks passed: True

| Split | Subset | Cases | Benign | High-grade |
|---|---|---:|---:|---:|
| 1 | train | 57 | 21 | 36 |
| 1 | val | 18 | 7 | 11 |
| 1 | test | 18 | 7 | 11 |
| 2 | train | 56 | 21 | 35 |
| 2 | val | 18 | 7 | 11 |
| 2 | test | 19 | 7 | 12 |
| 3 | train | 55 | 21 | 34 |
| 3 | val | 19 | 7 | 12 |
| 3 | test | 19 | 7 | 12 |
| 4 | train | 55 | 21 | 34 |
| 4 | val | 19 | 7 | 12 |
| 4 | test | 19 | 7 | 12 |
| 5 | train | 56 | 21 | 35 |
| 5 | val | 19 | 7 | 12 |
| 5 | test | 18 | 7 | 11 |

Checks:

- PASS: exactly_five_split_files
- PASS: split_1_no_duplicate_case_ids
- PASS: split_1_covers_all_eligible_cases
- PASS: split_1_disjoint_case_ids
- PASS: split_1_disjoint_case_families
- PASS: split_1_train_cases_both_classes
- PASS: split_1_train_cases_class_balance
- PASS: split_1_val_cases_both_classes
- PASS: split_1_val_cases_class_balance
- PASS: split_1_test_cases_both_classes
- PASS: split_1_test_cases_class_balance
- PASS: split_2_no_duplicate_case_ids
- PASS: split_2_covers_all_eligible_cases
- PASS: split_2_disjoint_case_ids
- PASS: split_2_disjoint_case_families
- PASS: split_2_train_cases_both_classes
- PASS: split_2_train_cases_class_balance
- PASS: split_2_val_cases_both_classes
- PASS: split_2_val_cases_class_balance
- PASS: split_2_test_cases_both_classes
- PASS: split_2_test_cases_class_balance
- PASS: split_3_no_duplicate_case_ids
- PASS: split_3_covers_all_eligible_cases
- PASS: split_3_disjoint_case_ids
- PASS: split_3_disjoint_case_families
- PASS: split_3_train_cases_both_classes
- PASS: split_3_train_cases_class_balance
- PASS: split_3_val_cases_both_classes
- PASS: split_3_val_cases_class_balance
- PASS: split_3_test_cases_both_classes
- PASS: split_3_test_cases_class_balance
- PASS: split_4_no_duplicate_case_ids
- PASS: split_4_covers_all_eligible_cases
- PASS: split_4_disjoint_case_ids
- PASS: split_4_disjoint_case_families
- PASS: split_4_train_cases_both_classes
- PASS: split_4_train_cases_class_balance
- PASS: split_4_val_cases_both_classes
- PASS: split_4_val_cases_class_balance
- PASS: split_4_test_cases_both_classes
- PASS: split_4_test_cases_class_balance
- PASS: split_5_no_duplicate_case_ids
- PASS: split_5_covers_all_eligible_cases
- PASS: split_5_disjoint_case_ids
- PASS: split_5_disjoint_case_families
- PASS: split_5_train_cases_both_classes
- PASS: split_5_train_cases_class_balance
- PASS: split_5_val_cases_both_classes
- PASS: split_5_val_cases_class_balance
- PASS: split_5_test_cases_both_classes
- PASS: split_5_test_cases_class_balance
- PASS: every_eligible_case_tested_exactly_once
- PASS: same_seed_reproduces_saved_splits
- PASS: different_seed_changes_test_partition
- PASS: five_reference_splits_available
- PASS: test_fold_partition_differs_from_reference
- PASS: split_1_not_duplicate_of_any_reference_assignment
- PASS: split_2_not_duplicate_of_any_reference_assignment
- PASS: split_3_not_duplicate_of_any_reference_assignment
- PASS: split_4_not_duplicate_of_any_reference_assignment
- PASS: split_5_not_duplicate_of_any_reference_assignment
