import os

DATASET_DEFAULT_CONFIG = {
    "THEIA_E5": {
        "raw_dir": "",
        "database": "theia_e5",
        "database_all_file": "theia_e5",
        "num_node_types": 3,
        "num_edge_types": 10,
        "timezone": "US/Eastern",
        "start_date": "2019-05-08",
        "end_date": "2019-05-18",
        "train_dates": ["2019-05-08", "2019-05-09", "2019-05-10"],
        "val_dates": ["2019-05-11"],
        "test_dates": ["2019-05-14", "2019-05-15"],
        "unused_dates": ["2019-05-12", "2019-05-13", "2019-05-16", "2019-05-17"],
        "ground_truth_relative_path": [
            "E5-THEIA/node_THEIA_1_Firefox_Drakon_APT_BinFmt_Elevate_Inject.csv"
        ],
        "attack_to_time_window": [
            [
                "E5-THEIA/node_THEIA_1_Firefox_Drakon_APT_BinFmt_Elevate_Inject.csv",
                "2019-05-15 14:47:00",
                "2019-05-15 15:08:00",
            ],
        ],
    },
    "THEIA_E3": {
        "raw_dir": "",
        "database": "theia_e3",
        "database_all_file": "theia_e3",
        "num_node_types": 3,
        "num_edge_types": 10,
        "timezone": "US/Eastern",
        "start_date": "2018-04-02",
        "end_date": "2018-04-14",
        "train_dates": [
            "2018-04-02",
            "2018-04-03",
            "2018-04-04",
            "2018-04-05",
            "2018-04-06",
            "2018-04-07",
            "2018-04-08",
        ],
        "val_dates": ["2018-04-09"],
        "test_dates": ["2018-04-10", "2018-04-12", "2018-04-13"],
        "unused_dates": ["2018-04-11"],
        "ground_truth_relative_path": [
            "E3-THEIA/node_Browser_Extension_Drakon_Dropper.csv",
            "E3-THEIA/node_Firefox_Backdoor_Drakon_In_Memory.csv",
            # "E3-THEIA/node_Phishing_E_mail_Executable_Attachment.csv", # attack failed so we don't use it
            # "E3-THEIA/node_Phishing_E_mail_Link.csv" # attack only at network level, not system
        ],
        "attack_to_time_window": [
            [
                "E3-THEIA/node_Browser_Extension_Drakon_Dropper.csv",
                "2018-04-12 12:40:00",
                "2018-04-12 13:30:00",
            ],
            [
                "E3-THEIA/node_Firefox_Backdoor_Drakon_In_Memory.csv",
                "2018-04-10 14:30:00",
                "2018-04-10 15:00:00",
            ],
        ],
    },
    "CADETS_E5": {
        "raw_dir": "",
        "database": "cadets_e5",
        "database_all_file": "cadets_e5",
        "num_node_types": 3,
        "num_edge_types": 10,
        "timezone": "US/Eastern",
        "start_date": "2019-05-08",
        "end_date": "2019-05-18",
        "train_dates": ["2019-05-08", "2019-05-09", "2019-05-11"],
        "val_dates": ["2019-05-12"],
        "test_dates": ["2019-05-16", "2019-05-17"],
        "unused_dates": ["2019-05-15", "2019-05-10", "2019-05-13", "2019-05-14"],
        "ground_truth_relative_path": [
            "E5-CADETS/node_Nginx_Drakon_APT.csv",
            "E5-CADETS/node_Nginx_Drakon_APT_17.csv",
        ],
        "attack_to_time_window": [
            ["E5-CADETS/node_Nginx_Drakon_APT.csv", "2019-05-16 09:31:00", "2019-05-16 10:12:00"],
            [
                "E5-CADETS/node_Nginx_Drakon_APT_17.csv",
                "2019-05-17 10:15:00",
                "2019-05-17 15:33:00",
            ],
        ],
    },
    "CADETS_E3": {
        "raw_dir": "",
        "database": "cadets_e3",
        "database_all_file": "cadets_e3",
        "num_node_types": 3,
        "num_edge_types": 10,
        "timezone": "US/Eastern",
        "start_date": "2018-04-02",
        "end_date": "2018-04-14",
        "train_dates": [
            "2018-04-02",
            "2018-04-03",
            "2018-04-04",
            "2018-04-05",
            "2018-04-07",
            "2018-04-08",
            "2018-04-09",
        ],
        "val_dates": ["2018-04-10"],
        "test_dates": ["2018-04-06", "2018-04-11", "2018-04-12", "2018-04-13"],
        "unused_dates": [],
        "ground_truth_relative_path": [
            # "E3-CADETS/node_E_mail_Server.csv",
            "E3-CADETS/node_Nginx_Backdoor_06.csv",
            # "E3-CADETS/node_Nginx_Backdoor_11.csv",
            "E3-CADETS/node_Nginx_Backdoor_12.csv",
            "E3-CADETS/node_Nginx_Backdoor_13.csv",
        ],
        "attack_to_time_window": [
            ["E3-CADETS/node_Nginx_Backdoor_06.csv", "2018-04-06 11:20:00", "2018-04-06 12:09:00"],
            # ["E3-CADETS/node_Nginx_Backdoor_11.csv" , '2018-04-11 15:07:00', '2018-04-11 15:16:00'],
            ["E3-CADETS/node_Nginx_Backdoor_12.csv", "2018-04-12 13:59:00", "2018-04-12 14:39:00"],
            ["E3-CADETS/node_Nginx_Backdoor_13.csv", "2018-04-13 09:03:00", "2018-04-13 09:16:00"],
        ],
    },
    "CLEARSCOPE_E5": {
        "raw_dir": "",
        "database": "clearscope_e5",
        "database_all_file": "clearscope_e5",
        "num_node_types": 3,
        "num_edge_types": 10,
        "timezone": "US/Eastern",
        "start_date": "2019-05-08",
        "end_date": "2019-05-18",
        "train_dates": ["2019-05-08", "2019-05-09", "2019-05-10", "2019-05-11", "2019-05-12"],
        "val_dates": ["2019-05-13"],
        "test_dates": ["2019-05-14", "2019-05-15", "2019-05-17"],
        "unused_dates": ["2019-05-16"],
        "ground_truth_relative_path": [
            "E5-CLEARSCOPE/node_clearscope_e5_appstarter_0515.csv",
            # "E5-CLEARSCOPE/node_clearscope_e5_firefox_0517.csv",
            # "E5-CLEARSCOPE/node_clearscope_e5_lockwatch_0517.csv",
            "E5-CLEARSCOPE/node_clearscope_e5_tester_0517.csv",
        ],
        "attack_to_time_window": [
            [
                "E5-CLEARSCOPE/node_clearscope_e5_appstarter_0515.csv",
                "2019-05-15 15:38:00",
                "2019-05-15 16:19:00",
            ],
            # ["E5-CLEARSCOPE/node_clearscope_e5_firefox_0517.csv", '2019-05-17 11:49:00', '2019-05-17 15:32:00'],
            [
                "E5-CLEARSCOPE/node_clearscope_e5_lockwatch_0517.csv",
                "2019-05-17 15:48:00",
                "2019-05-17 16:01:00",
            ],
            [
                "E5-CLEARSCOPE/node_clearscope_e5_tester_0517.csv",
                "2019-05-17 16:20:00",
                "2019-05-17 16:28:00",
            ],
        ],
    },
    "CLEARSCOPE_E3": {
        "raw_dir": "",
        "database": "clearscope_e3",
        "database_all_file": "clearscope_e3",
        "num_node_types": 3,
        "num_edge_types": 10,
        "timezone": "US/Eastern",
        "start_date": "2018-04-02",
        "end_date": "2018-04-14",
        "train_dates": [
            "2018-04-03",
            "2018-04-04",
            "2018-04-05",
            "2018-04-07",
            "2018-04-08",
            "2018-04-09",
            "2018-04-10",
        ],
        "val_dates": ["2018-04-02"],
        "test_dates": ["2018-04-11", "2018-04-12"],
        "unused_dates": ["2018-04-06", "2018-04-13"],
        "ground_truth_relative_path": [
            "E3-CLEARSCOPE/node_clearscope_e3_firefox_0411.csv",
            # "E3-CLEARSCOPE/node_clearscope_e3_firefox_0412.csv", # due to malicious file downloaded but failed to exec and feture missing, there is no malicious nodes found in database
        ],
        "attack_to_time_window": [
            [
                "E3-CLEARSCOPE/node_clearscope_e3_firefox_0411.csv",
                "2018-04-11 13:54:00",
                "2018-04-11 14:48:00",
            ],
            # ["E3-CLEARSCOPE/node_clearscope_e3_firefox_0412.csv", '2018-04-12 15:18:00', '2018-04-12 15:25:00'],
        ],
    },
    "optc_h201": {
        "raw_dir": "",
        "database": "optc_201",
        "database_all_file": "optc_201",
        "num_node_types": 3,
        "num_edge_types": 10,
        "timezone": "Etc/GMT+4",
        "start_date": "2019-09-15",
        "end_date": "2019-09-26",
        "train_dates": ["2019-09-19", "2019-09-20", "2019-09-21"],
        "val_dates": ["2019-09-22"],
        "test_dates": ["2019-09-23", "2019-09-24", "2019-09-25"],
        "unused_dates": ["2019-09-16", "2019-09-17", "2019-09-18"],
        "ground_truth_relative_path": [
            "h201/node_h201_0923.csv",
        ],
        "attack_to_time_window": [
            ["h201/node_h201_0923.csv", "2019-09-23 11:23:00", "2019-09-23 13:25:00"],
        ],
    },
    "optc_h501": {
        "raw_dir": "",
        "database": "optc_501",
        "database_all_file": "optc_501",
        "num_node_types": 3,
        "num_edge_types": 10,
        "timezone": "Etc/GMT+4",
        "start_date": "2019-09-15",
        "end_date": "2019-09-26",
        "train_dates": ["2019-09-19", "2019-09-20", "2019-09-21"],
        "val_dates": ["2019-09-22"],
        "test_dates": ["2019-09-23", "2019-09-24", "2019-09-25"],
        "unused_dates": ["2019-09-16", "2019-09-17", "2019-09-18"],
        "ground_truth_relative_path": [
            "h501/node_h501_0924.csv",
        ],
        "attack_to_time_window": [
            ["h501/node_h501_0924.csv", "2019-09-24 10:28:00", "2019-09-24 15:29:00"],
        ],
    },
    "optc_h051": {
        "raw_dir": "",
        "database": "optc_051",
        "database_all_file": "optc_051",
        "num_node_types": 3,
        "num_edge_types": 10,
        "timezone": "Etc/GMT+4",
        "start_date": "2019-09-15",
        "end_date": "2019-09-26",
        "train_dates": ["2019-09-19", "2019-09-20", "2019-09-21"],
        "val_dates": ["2019-09-22"],
        "test_dates": ["2019-09-23", "2019-09-24", "2019-09-25"],
        "unused_dates": ["2019-09-16", "2019-09-17", "2019-09-18"],
        "ground_truth_relative_path": [
            "h051/node_h051_0925.csv",
        ],
        "attack_to_time_window": [
            ["h051/node_h051_0925.csv", "2019-09-25 10:29:00", "2019-09-25 14:25:00"],
        ],
    },
    "TRACE_E5": {
        "raw_dir": "",
        "database": "trace_e5",
        "database_all_file": "trace_e5",
        "num_node_types": 3,
        "num_edge_types": 10,
        "timezone": "US/Eastern",
        "start_date": "2019-05-08",
        "end_date": "2019-05-18",
        "train_dates": ["2019-05-08", "2019-05-09", "2019-05-10", "2019-05-11", "2019-05-12"],
        "val_dates": ["2019-05-13"],
        "test_dates": ["2019-05-14", "2019-05-15"],
        "unused_dates": ["2019-05-16", "2019-05-17"],
        "ground_truth_relative_path": [
            "E5-TRACE/node_Trace_Firefox_Drakon.csv",
        ],
        "attack_to_time_window": [
            [
                "E5-TRACE/node_Trace_Firefox_Drakon.csv",
                "2019-05-14 10:17:00",
                "2019-05-14 11:45:00",
            ],
        ],
    },
    "TRACE_E3": {
        "raw_dir": "",
        "database": "trace_e3",
        "database_all_file": "trace_e3",
        "num_node_types": 3,
        "num_edge_types": 10,
        "timezone": "US/Eastern",
        "start_date": "2018-04-02",
        "end_date": "2018-04-14",
        "train_dates": [
            "2018-04-02",
            "2018-04-03",
            "2018-04-04",
            "2018-04-05",
            "2018-04-06",
            "2018-04-07",
            "2018-04-08",
        ],
        "val_dates": ["2018-04-09",],
        "test_dates": [
            "2018-04-10",
            "2018-04-11",
            "2018-04-12",
            "2018-04-13",
        ],
        "unused_dates": [],
        "ground_truth_relative_path": [
            "E3-TRACE/node_trace_e3_firefox_0410.csv",
            "E3-TRACE/node_trace_e3_phishing_executable_0413.csv",
            "E3-TRACE/node_trace_e3_pine_0413.csv",
        ],
        "attack_to_time_window": [
            [
                "E3-TRACE/node_trace_e3_firefox_0410.csv",
                "2018-04-10 09:45:00",
                "2018-04-10 11:10:00",
            ],
            [
                "E3-TRACE/node_trace_e3_phishing_executable_0413.csv",
                "2018-04-13 14:14:00",
                "2018-04-13 14:29:00",
            ],
            ["E3-TRACE/node_trace_e3_pine_0413.csv", "2018-04-13 12:42:00", "2018-04-13 12:54:00"],
        ],
    },
    "FIVEDIRECTIONS_E5": {
        "raw_dir": "",
        "database": "fivedirections_e5",
        "database_all_file": "fivedirections_e5",
        "num_node_types": 3,
        "num_edge_types": 10,
        "timezone": "US/Eastern",
        "start_date": "2019-05-08",
        "end_date": "2019-05-18",
        "train_dates": ["2019-05-08", "2019-05-10", "2019-05-11", "2019-05-13", "2019-05-14"],
        "val_dates": ["2019-05-12"],
        "test_dates": [
            "2019-05-09",
            "2019-05-15",
            "2019-05-17",
        ],
        "unused_dates": ["2019-05-16"],
        "ground_truth_relative_path": [
            "E5-FIVEDIRECTIONS/node_fivedirections_e5_bits_0515.csv",
            "E5-FIVEDIRECTIONS/node_fivedirections_e5_copykatz_0509.csv",
            "E5-FIVEDIRECTIONS/node_fivedirections_e5_dns_0517.csv",
            "E5-FIVEDIRECTIONS/node_fivedirections_e5_drakon_0517.csv",
        ],
        "attack_to_time_window": [
            [
                "E5-FIVEDIRECTIONS/node_fivedirections_e5_bits_0515.csv",
                "2019-05-15 13:14:00",
                "2019-05-15 13:35:00",
            ],
            [
                "E5-FIVEDIRECTIONS/node_fivedirections_e5_copykatz_0509.csv",
                "2019-05-09 13:25:00",
                "2019-05-09 13:57:00",
            ],
            [
                "E5-FIVEDIRECTIONS/node_fivedirections_e5_dns_0517.csv",
                "2019-05-17 12:46:00",
                "2019-05-17 12:57:00",
            ],
            [
                "E5-FIVEDIRECTIONS/node_fivedirections_e5_drakon_0517.csv",
                "2019-05-17 16:10:00",
                "2019-05-17 16:16:00",
            ],
        ],
    },
    "FIVEDIRECTIONS_E3": {
        "raw_dir": "",
        "database": "fivedirections_e3",
        "database_all_file": "fivedirections_e3",
        "num_node_types": 3,
        "num_edge_types": 10,
        "timezone": "US/Eastern",
        "start_date": "2018-04-02",
        "end_date": "2018-04-14",
        "train_dates": [
            "2018-04-03",
            "2018-04-05",
            "2018-04-06",
            "2018-04-07",
            "2018-04-08",
            "2018-04-10",
            "2018-04-13",
        ],
        "val_dates": ["2018-04-04"],
        "test_dates": ["2018-04-09", "2018-04-11"],
        "unused_dates": ["2018-04-12"],
        "ground_truth_relative_path": [
            "E3-FIVEDIRECTIONS/node_fivedirections_e3_firefox_0411.csv",
            # "E3-FIVEDIRECTIONS/node_fivedirections_e3_browser_0412.csv",
            "E3-FIVEDIRECTIONS/node_fivedirections_e3_excel_0409.csv",
        ],
        "attack_to_time_window": [
            [
                "E3-FIVEDIRECTIONS/node_fivedirections_e3_firefox_0411.csv",
                "2018-04-11 09:59:00",
                "2018-04-11 10:41:00",
            ],
            # ["E3-FIVEDIRECTIONS/node_fivedirections_e3_browser_0412.csv", '2018-04-12 11:12:00', '2018-04-12 11:15:00'],
            [
                "E3-FIVEDIRECTIONS/node_fivedirections_e3_excel_0409.csv",
                "2018-04-09 15:06:00",
                "2018-04-09 15:43:00",
            ],
        ],
    },
    "PROVENANCE_BENIGN": {
        "raw_dir": "",
        "database": "PROVENANCE_BENIGN",
        "database_all_file": "PROVENANCE_BENIGN",
        "num_node_types": 3,
        "num_edge_types": 10,
        "timezone": "UTC",
        "start_date": "2026-03-16",
        "end_date": "2026-03-16",
        "train_dates": ["2026-03-16"],
        "val_dates": ["2026-03-16"],
        "test_dates": ["2026-03-16"],
        "unused_dates": [],
        "ground_truth_relative_path": [],
        "attack_to_time_window": [],
    },
    # See https://arxiv.org/pdf/2401.01341
    "ATLASV2_EDR": {
        "raw_dir": "",
        "database": "atlasv2_edr",
        "database_all_file": "atlasv2_edr",
        "num_node_types": 3,
        "num_edge_types": 33,
        "timezone": "US/Central",
        "start_date": "2022-07-15",
        "end_date": "2022-07-21",
        "train_dates": [
            "2022-07-16",
            # Arbitrarly picked 2022-07-17 for the validation/threshold calibration
            "2022-07-18",
        ],
        "val_dates": [
            "2022-07-17"
        ],
        "test_dates": [
            "2022-07-19",
            "2022-07-20"
        ],
        "unused_dates": [
            "2022-07-15"
        ],
        "ground_truth_relative_path": [
            "atlasv2_edr/atlasv2_edr_s1.csv",
            "atlasv2_edr/atlasv2_edr_s2.csv",
            "atlasv2_edr/atlasv2_edr_s3.csv",
            "atlasv2_edr/atlasv2_edr_s4.csv",
            "atlasv2_edr/atlasv2_edr_m1.csv",
            "atlasv2_edr/atlasv2_edr_m2.csv",
            "atlasv2_edr/atlasv2_edr_m3.csv",
            "atlasv2_edr/atlasv2_edr_m4.csv",
            "atlasv2_edr/atlasv2_edr_m5.csv",
            "atlasv2_edr/atlasv2_edr_m6.csv",
        ],
        "attack_to_time_window": [
            # NOTE: the reported attack windows are somewhat inaccurate (i.e., the first and last
            # true-positive malicious alerts occur OUTSIDE the reported attack windows), so we start
            # at 2022-07-19 13:00:00 (Rather than 2022-07-19 13:12:00) and end at
            # 2022-07-20 01:15:00 (Rather than 2022-07-20 01:00:00) to capture all of the true
            # positives.
            ["atlasv2_edr/atlasv2_edr_s1.csv", "2022-07-19 13:00:00", "2022-07-19 13:40:00"],
            ["atlasv2_edr/atlasv2_edr_s2.csv", "2022-07-19 13:45:00", "2022-07-19 14:20:00"],
            ["atlasv2_edr/atlasv2_edr_s3.csv", "2022-07-19 14:20:00", "2022-07-19 15:05:00"],
            ["atlasv2_edr/atlasv2_edr_s4.csv", "2022-07-20 00:31:00", "2022-07-20 01:15:00"],
            ["atlasv2_edr/atlasv2_edr_m1.csv", "2022-07-19 16:00:00", "2022-07-19 17:50:00"],
            ["atlasv2_edr/atlasv2_edr_m2.csv", "2022-07-19 19:32:00", "2022-07-19 20:02:00"],
            ["atlasv2_edr/atlasv2_edr_m3.csv", "2022-07-19 20:06:00", "2022-07-19 20:40:00"],
            ["atlasv2_edr/atlasv2_edr_m4.csv", "2022-07-19 22:31:00", "2022-07-19 23:04:00"],
            ["atlasv2_edr/atlasv2_edr_m5.csv", "2022-07-19 23:16:00", "2022-07-19 23:46:00"],
            ["atlasv2_edr/atlasv2_edr_m6.csv", "2022-07-19 23:54:00", "2022-07-20 00:27:00"],
        ],
    },
    # See https://www.ndss-symposium.org/wp-content/uploads/prism2026-12.pdf
    "CARBANAKV2_EDR": {
        "raw_dir": "",
        "database": "carbanakv2_edr",
        "database_all_file": "carbanakv2_edr",
        "num_node_types": 3,
        "num_edge_types": 33,
        "timezone": "US/Central",
        "start_date": "2024-04-18",
        "end_date": "2024-05-13",
        "train_dates": [
            "2024-04-20",
            "2024-04-21",
            # Arbitrarly picked 2024-04-22 for the validation/threshold calibration
            "2024-04-23",
            "2024-04-24",
            "2024-04-25",
            # Arbitrarly picked 2024-04-26 for the validation/threshold calibration
            "2024-04-27",
            "2024-04-28",
            "2024-04-29",
        ],
        "val_dates": [
            "2024-04-22",
            "2024-04-26",
        ],
        "test_dates": [
            "2024-04-30",
            "2024-05-01",
            "2024-05-02",
            "2024-05-07",
            "2024-05-08",
            "2024-05-09",
            "2024-05-10",
        ],
        "unused_dates": [
            "2024-04-18",
            "2024-04-19",
            "2024-05-03",
            "2024-05-04",
            "2024-05-05",
            "2024-05-06",
            "2024-05-11",
            "2024-05-12",
            "2024-05-13",
        ],
        "ground_truth_relative_path": [
            "carbanakv2_edr/carbanakv2_edr.csv",
        ],
        "attack_to_time_window": [
            ["carbanakv2_edr/carbanakv2_edr.csv", "2024-04-30 17:30:00", "2024-05-10 20:30:00"]
        ],
    },
}

# Arguments

TASK_DEPENDENCIES = {
    "construction": [],
    "transformation": ["construction"],
    "featurization": ["transformation"],
    "feat_inference": ["featurization"],
    "batching": ["feat_inference"],
    "training": ["batching"],
    "evaluation": ["training"],
    "triage": ["evaluation"],
}


class AND(list):
    pass


class OR(list):
    pass


class Arg:
    def __init__(self, type, vals: list = None, desc: str = None):
        self.type = type
        self.vals = vals
        self.desc = desc


FEATURIZATIONS_CFG = {
    "word2vec": {
        "alpha": Arg(float),
        "window_size": Arg(int),
        "min_count": Arg(int),
        "use_skip_gram": Arg(bool),
        "num_workers": Arg(int),
        "epochs": Arg(int),
        "compute_loss": Arg(bool),
        "negative": Arg(int),
        "decline_rate": Arg(int),
    },
    "doc2vec": {
        "include_neighbors": Arg(bool),
        "epochs": Arg(int),
        "alpha": Arg(float),
    },
    "fasttext": {
        "min_count": Arg(int),
        "alpha": Arg(float),
        "window_size": Arg(int),
        "negative": Arg(int),
        "num_workers": Arg(int),
        "use_pretrained_fb_model": Arg(bool),
    },
    "alacarte": {
        "walk_length": Arg(int),
        "num_walks": Arg(int),
        "epochs": Arg(int),
        "context_window_size": Arg(int),
        "min_count": Arg(int),
        "use_skip_gram": Arg(bool),
        "num_workers": Arg(int),
        "compute_loss": Arg(bool),
        "add_paths": Arg(bool),
    },
    "temporal_rw": {
        "walk_length": Arg(int),
        "num_walks": Arg(int),
        "trw_workers": Arg(int),
        "time_weight": Arg(str),
        "half_life": Arg(int),
        "window_size": Arg(int),
        "min_count": Arg(int),
        "use_skip_gram": Arg(bool),
        "wv_workers": Arg(int),
        "epochs": Arg(int),
        "compute_loss": Arg(bool),
        "negative": Arg(int),
        "decline_rate": Arg(int),
    },
    "flash": {
        "min_count": Arg(int),
        "workers": Arg(int),
    },
    "hierarchical_hashing": {},
    "magic": {},
    "only_type": {},
    "only_ones": {},
    "ocrapt_features": {
        "use_lifespan": Arg(bool, desc="Off by default, hurts generalization (paper Appendix E)."),
        "use_cumulative_active_time": Arg(bool, desc="Off by default, same reason as use_lifespan."),
    },
    "spider": {
        # Model configuration
        "model_size": Arg(
            str,
            desc="SPIDER encoder size preset (selects hidden / layers / heads / FFN). "
                 "Standard presets: 'tiny' (H=128), 'mini' (H=256), 'med' (H=512), "
                 "'baseline' (H=768). HF pretrained model_types use their own size keys "
                 "(e.g. 'small' / 'medium' / 'large' / 'xl' for gpt2_pretrained, "
                 "'1b' / '3b' for llama3_pretrained).",
        ),
        "model_type": Arg(
            str,
            vals=OR([
                # MLM walk-based
                "bert", "roberta", "modernbert", "ropebert", "llama", "logbert",
                # Walk embedding
                "deepwalk", "node2vec",
                # GNN self-supervised
                "graphmae", "gae", "dgi",
                # GNN token-budget
                "gnn_distill", "behavior_cluster", "spider",
                # HF pretrained
                "gpt2_pretrained", "llama3_pretrained", "opt_pretrained",
            ]),
            desc=(
                "SPIDER pretraining objective + architecture. See "
                "config/pretrained/README.md for full descriptions. Families:\n"
                "  MLM walk-based     : bert, roberta, modernbert, ropebert, llama, logbert\n"
                "  Walk embedding     : deepwalk, node2vec\n"
                "  GNN self-supervised: graphmae, gae, dgi\n"
                "  GNN token-budget   : gnn_distill, behavior_cluster, spider\n"
                "  HF pretrained      : gpt2_pretrained, llama3_pretrained, opt_pretrained"
            ),
        ),
        # DeepWalk / Node2Vec shared parameters
        "deepwalk": {
            "window": Arg(int, desc="Skip-gram context window size. Default: 5."),
            "epochs": Arg(int, desc="Number of Word2Vec training epochs. Default: 10."),
            "min_count": Arg(int, desc="Minimum label frequency to include in vocabulary. Default: 1."),
            "workers": Arg(int, desc="Number of parallel workers for Word2Vec training. Default: 4."),
        },

        # Node2Vec-specific parameters
        "node2vec": {
            "p": Arg(float, desc="Return parameter. Higher p = less likely to revisit previous node. Default: 1.0."),
            "q": Arg(float, desc="In-out parameter. q > 1 = BFS-like (local); q < 1 = DFS-like (explore). Default: 1.0."),
        },

        # GNN distillation model_type parameters
        "gnn_distill": {
            "emb_dim": Arg(int, desc="GNN edge embedding dimension. Default: 128."),
            "hidden_dim": Arg(int, desc="GNN hidden dimension. Default: 128."),
            "num_layers": Arg(int, desc="Number of GNN encoder layers. Default: 1."),
            "num_heads": Arg(int, desc="Number of attention heads in GNN layers. Default: 4."),
            "n_neighbors_min": Arg(int, desc="Min direct neighbors to sample per entity. Default: 5."),
            "n_neighbors_max": Arg(int, desc="Max direct neighbors to sample per entity. Default: 20."),
            "diverse_neighbors": Arg(bool, desc="Dedup neighbors by (edge_type, node_type, label) and prioritize edge type diversity. Default: True."),
            "loss_weight": Arg(float, desc="Weight for distillation loss (T5 encoder → GNN alignment). Default: 0.5."),
            "gnn_loss_weight": Arg(float, desc="Weight for GNN masked reconstruction loss. Default: 1.0."),
            "gnn_lr": Arg(float, desc="Learning rate for GNN teacher parameters. Default: 1e-3."),
            "ema_momentum": Arg(float, desc="EMA momentum for teacher update (0.999 = slow-moving). Default: 0.999."),
            "edge_projection": Arg(bool, desc="Apply per-edge-type linear projection to source nodes before GNN message passing. Default: False."),
            "filter_noisy_edges": Arg(bool, desc="Remove shared libs, /dev, /proc, /sys, linker cache, common configs from GNN neighborhoods. Default: False."),
        },

        # Behavior cluster model_type parameters
        "behavior_cluster": {
            "proj_dim": Arg(int, desc="Contrastive projection head output dimensionality. Default: 128."),
            "bce_weight": Arg(float, desc="Weight for multi-label BCE loss. Default: 1.0."),
            "contrastive_weight": Arg(float, desc="Weight for contrastive (NT-Xent) loss. Default: 0.5."),
            "temperature": Arg(float, desc="Temperature for InfoNCE cosine similarity scaling. Default: 0.07."),
            "min_signature_size": Arg(int, desc="Skip entities with fewer behavior labels in their signature. Default: 2."),
            "filter_noisy_edges": Arg(bool, desc="Remove shared libs, /dev, /proc, /sys, linker cache, common configs from signature extraction. Default: True."),
            "use_contrastive_head": Arg(bool, desc="Use contrastive projection head for inference embeddings (proj_dim) instead of raw encoder (emb_dim). Default: False."),
            "min_entities_per_class": Arg(int, desc="Floor: oversample small classes to this minimum per epoch. 0 = no floor. Default: 0."),
            "max_entities_per_class": Arg(int, desc="Cap entities per signature class per epoch. Rotates across epochs. 0 = no cap. Default: 0."),
            "samples_per_class": Arg(int, desc="K in P×K batch sampling: number of entities per class per batch. Guarantees every entity has at least K-1 positives for contrastive learning. Default: 4."),
            "bce_mode": Arg(str, desc="Classification head mode: 'multilabel' = BCE over behavior labels, 'class' = cross-entropy over entity classes. Default: multilabel."),
            "contrastive_target": Arg(str, desc="Contrastive positive definition: 'signature' = entities with identical behavior signatures are positives, 'entity_class' = entities with the same coarse functional class are positives. Default: signature."),
            "strip_entity_type": Arg(bool, desc="Remove entity type prefix tokens ([PROC], [FILE], [SOCK]) from tokenized sequences. Forces the model to learn identity from behavior rather than type. Default: False."),
        },

        # GNN cluster model_type parameters
        "spider": {
            "emb_dim": Arg(int, desc="GNN edge embedding dimension. Default: 256."),
            "hidden_dim": Arg(int, desc="GNN encoder hidden dimension. Default: 256."),
            "proj_dim": Arg(int, desc="Contrastive projection dimension (teacher). Default: 256."),
            "num_heads": Arg(int, desc="Number of attention heads in GNN TransformerConv. Default: 4."),
            "n_neighbors_min": Arg(int, desc="Min direct neighbors to sample per entity. Default: 5."),
            "n_neighbors_max": Arg(int, desc="Max direct neighbors to sample per entity. Default: 20."),
            "diverse_neighbors": Arg(bool, desc="Dedup neighbors by (edge_type, node_type, label) and prioritize edge type diversity. Default: True."),
            "filter_noisy_edges": Arg(bool, desc="Remove shared libs, /dev, /proc, /sys from GNN neighborhoods. Default: True."),
            "ema_momentum": Arg(float, desc="EMA momentum for teacher T5 update. Default: 0.999."),
            "supcon_weight": Arg(float, desc="Weight for supervised contrastive loss on GNN teacher. Default: 1.0."),
            "distill_weight": Arg(float, desc="Weight for distillation loss (student → GNN targets). Default: 0.5."),
            "temperature": Arg(float, desc="SupCon temperature. Default: 0.07."),
            "min_signature_size": Arg(int, desc="Skip entities with fewer behavior labels (for entity class assignment). Default: 2."),
            "min_entities_per_class": Arg(int, desc="Floor: oversample small classes to this minimum per epoch. Default: 100."),
            "max_entities_per_class": Arg(int, desc="Cap entities per class per epoch. Default: 2000."),
            "samples_per_class": Arg(int, desc="K in P×K batching. Default: 4."),
            "strip_entity_type": Arg(bool, desc="Remove entity type prefix tokens from inputs. Default: False."),
            "distill_loss": Arg(str, desc="Distillation loss: sce or mse. Default: sce.", vals=OR(["sce", "mse"])),
            "teacher_loss": Arg(str, desc="Teacher loss: contrastive (SupCon) or bce (cross-entropy). Default: contrastive.", vals=OR(["contrastive", "bce"])),
            "teacher_data": Arg(str, desc="Teacher data modalities (comma-separated): signature, gnn_emb, or both. Default: signature,gnn_emb."),
            "student_only_mode": Arg(str, desc="Student-only ablation: none, student_signature, or student_class. Default: none.", vals=OR(["none", "student_signature", "student_class"])),
        },

        # GPT-2 pretrained parameters
        "gpt2_pretrained": {
            "max_seq_len": Arg(int, desc="Max sequence length for HF tokenizer. Default: 512."),
        },

        # Llama 3.2 pretrained parameters
        "llama3_pretrained": {
            "max_seq_len": Arg(int, desc="Max sequence length for HF tokenizer. Default: 512."),
        },

        # OPT pretrained parameters
        "opt_pretrained": {
            "max_seq_len": Arg(int, desc="Max sequence length for HF tokenizer. Default: 512."),
        },

        # GraphMAE-specific parameters
        "graphmae": {
            "num_layers": Arg(int, desc="Number of GAT encoder layers. Default: 2."),
            "num_heads": Arg(int, desc="Number of attention heads in GAT layers. Default: 4."),
            "decoder_num_layers": Arg(int, desc="Number of GAT decoder layers. Default: 1."),
            "mask_rate": Arg(float, desc="Fraction of nodes to mask during pretraining. Default: 0.5."),
            "replace_rate": Arg(float, desc="Fraction of masked nodes replaced with random tokens (rest get [MASK]). Default: 0.1."),
            "neighborhood_min": Arg(int, desc="Minimum number of neighbors in temporal neighborhood. Default: 5."),
            "neighborhood_max": Arg(int, desc="Maximum number of neighbors in temporal neighborhood. Default: 20."),
            "epochs": Arg(int, desc="Number of training epochs. Default: 100."),
            "lr": Arg(float, desc="Learning rate. Default: 0.001."),
        },

        # GAE-specific parameters
        "gae": {
            "num_layers": Arg(int, desc="Number of GAT encoder layers. Default: 2."),
            "num_heads": Arg(int, desc="Number of attention heads in GAT layers. Default: 4."),
            "neighborhood_min": Arg(int, desc="Minimum number of neighbors in temporal neighborhood. Default: 5."),
            "neighborhood_max": Arg(int, desc="Maximum number of neighbors in temporal neighborhood. Default: 20."),
            "epochs": Arg(int, desc="Number of training epochs. Default: 100."),
            "lr": Arg(float, desc="Learning rate. Default: 0.001."),
        },

        # DGI-specific parameters
        "dgi": {
            "num_layers": Arg(int, desc="Number of GAT encoder layers. Default: 2."),
            "num_heads": Arg(int, desc="Number of attention heads in GAT layers. Default: 4."),
            "neighborhood_min": Arg(int, desc="Minimum number of neighbors in temporal neighborhood. Default: 5."),
            "neighborhood_max": Arg(int, desc="Maximum number of neighbors in temporal neighborhood. Default: 20."),
            "epochs": Arg(int, desc="Number of training epochs. Default: 100."),
            "lr": Arg(float, desc="Learning rate. Default: 0.001."),
        },

        # Token-budget training loop. Used by: MLM family, GNN token-budget
        # (gnn_distill / spider / behavior_cluster), HF pretrained. Ignored by:
        # walk embedding (uses deepwalk.epochs) and GNN-SSL (graphmae / gae / dgi
        # have their own .epochs and .lr). See spider.yml for the full scope map.
        "training": {
            "pretrain_tokens": Arg(int, desc="Total number of tokens to process during pretraining (controls training duration)."),
            "warmup_tokens": Arg(int, desc="Number of tokens for learning rate warmup at the start of pretraining."),
            "batch_size": Arg(int, desc="Mini-batch size during pretraining. Interpretation depends on model_type: walks per batch (MLM), entities per batch (GNN token-budget), labels per batch (HF pretrained), neighborhoods per batch (GNN-SSL)."),
            "lr": Arg(float, desc="Peak learning rate for pretraining (after warmup)."),
            "scheduler": Arg(str, vals=OR(["cosine", "linear"]), desc="Learning rate scheduler shape after warmup. Only consumed by MLM and HF pretrained; GNN token-budget hardcodes cosine."),
        },

        # MLM-specific parameters (masking + architecture sub-blocks for MLM family)
        "mlm": {
            "mask_rate_fixed": Arg(float, desc="Initial fixed mask rate (fraction of nodes masked). Decays to mask_rate_min."),
            "mask_rate_min": Arg(float, desc="Minimum mask rate after decay."),
            "mask_edge_type": Arg(bool, desc="Enable structured masking of edge types during pretraining and edge-type scoring at inference."),

            # ModernBERT-specific parameters
            "modernbert": {
                "global_attn_every_n_layers": Arg(int, desc="Use global attention every N layers. Other layers use local sliding window attention. Default: 3."),
                "local_attention_window": Arg(int, desc="Sliding window size for local attention layers. Default: 128 tokens."),
            },

            # RoPEBERT-specific parameters
            "ropebert": {
                "rope_theta": Arg(float, desc="Base frequency for rotary position embeddings. Default: 10000.0."),
            },

            # LLaMA-specific parameters
            "llama": {
                "rope_theta": Arg(float, desc="Base frequency for rotary position embeddings. Default: 10000.0."),
            },

            # LogBERT-specific parameters
            "logbert": {
                "hvm_weight": Arg(float, desc="Weight for hypersphere volume minimization loss relative to MLM loss. Default: 0.1."),
            },
        },

        "pretrain_datasets": Arg(str, desc="Comma-separated list of dataset names to use for multi-dataset pretraining."),
        # Tokenizer configuration (affects tokenizer cache)
        "tokenizer": {
            "bpe_vocab_size": Arg(int, desc="Target BPE vocabulary size for the tokenizer."),
            "max_seq_len": Arg(int, desc="Maximum token sequence length after tokenization. Walks exceeding this are truncated."),
            "mode": Arg(str, vals=OR(["domain_bpe", "bpe_only"]), desc="Tokenizer mode: 'domain_bpe' uses domain-specific pre-tokenization followed by BPE; 'bpe_only' skips domain rules and applies BPE directly on raw words."),
            "normalize_netflow_ips": Arg(bool, desc="Replace IP addresses in netflow entities with category tokens ([PRIVATE_IP], [PUBLIC_IP], [LOCALHOST_IP]) instead of keeping individual octets."),
            "canonicalize_neighbors": Arg(bool, desc="Enable multi-level neighbor canonicalization. When True, decoder targets keep only structural/categorical tokens ([] special tokens + OS-agnostic category tokens like [CAT_WEBSERVER], [FCAT_LOG_WEB]) while encoder input gets full detail + category tokens prepended. During continue_pretrain, neighbors keep all tokens with category tokens prepended."),
        },
        "spider_path": Arg(str, desc="Path to a pretrained SPIDER model folder. May contain any subset of: corpus.pt (sampler states + indexid2msg), tokenizer.pt, pretrain_*.pt (model checkpoint), behavior_vocab.txt. Present artifacts are loaded; missing ones are computed from scratch. Works with any model_type."),

        "graph_context_mode": Arg(
            str,
            vals=OR(["window", "day", "all"]),
            desc="Graph context scope for walk sampling at both pretraining and inference time. "
                 "'window' = each time-window snapshot independently; "
                 "'day' = merge all snapshots from the same calendar day; "
                 "'all' = merge all snapshots in the relevant split "
                 "(train split at pretrain time; val or test split at inference time — "
                 "no training data leaks into inference context).",
        ),

        # Walk configuration (used by MLM family and deepwalk/node2vec; affects corpus cache)
        "walks": {
            "walk_length": Arg(int, desc="Number of nodes per random walk during pretraining."),
            "num_walks": Arg(int, desc="Number of random walks to sample per node per epoch during pretraining."),
            "time_weight": Arg(str, desc="Temporal weighting for neighbor selection: 'uniform', 'exponential', or 'linear'."),
            "half_life": Arg(float, desc="Half-life (in seconds) for exponential time weighting of neighbor selection."),
            "random_walk_start": Arg(
                bool,
                desc="Randomize the temporal entry point for each walk. "
                     "When True, the first hop picks a random edge instead of "
                     "always starting from the earliest/latest timestamp.",
            ),
            "diversity_weight": Arg(
                float,
                desc="Edge-type diversity bias for random walks during pretraining. "
                     "0 = no bias (default), higher values increasingly favor edges "
                     "whose type is underrepresented in the current walk. "
                     "Only affects pretraining; finetuning/inference use natural distribution.",
            ),
        },

    },
}

ENCODERS_CFG = {
    "tgn": {
        "tgn_memory_dim": Arg(int),
        "tgn_time_dim": Arg(int),
        "use_node_feats_in_gnn": Arg(bool),
        "use_memory": Arg(bool),
        "use_time_order_encoding": Arg(bool),
        "project_src_dst": Arg(bool),
        "mode": Arg(str),
        "use_residual_norm": Arg(bool, desc="Adds the projected input node features to the output of the GNN wrapped by TGN, followed by dropout and LayerNorm."),
    },
    "graph_attention": {
        "activation": Arg(str),
        "num_heads": Arg(int),
        "concat": Arg(bool),
        "flow": Arg(str),
        "num_layers": Arg(int),
    },
    "sage": {
        "activation": Arg(str),
        "num_layers": Arg(int),
    },
    "gat": {
        "activation": Arg(str),
        "num_heads": Arg(int),
        "concat": Arg(bool),
        "flow": Arg(str),
        "num_layers": Arg(int),
    },
    "gin": {
        "activation": Arg(str),
        "num_layers": Arg(int),
    },
    "rgcn": {
        "activation": Arg(str),
        "num_layers": Arg(int),
    },
    "rgcn_per_type": {
        "activation": Arg(str),
        "num_layers": Arg(int),
    },
    "sum_aggregation": {},
    "rcaid_gat": {},
    "magic_gat": {
        "num_layers": Arg(int),
        "num_heads": Arg(int),
        "negative_slope": Arg(float),
        "alpha_l": Arg(float),
        "activation": Arg(str),
    },
    "glstm": {},
    "custom_mlp": {
        "architecture_str": Arg(str),
    },
    "none": {},
    "hetero_graph_transformer": {
        "activation": Arg(str, desc="Unused: the input projections always use ReLU."),
        "num_heads": Arg(int, desc="Number of attention heads of each Heterogeneous Graph Transformer (HGT) layer."),
        "num_layers": Arg(int, desc="Number of HGT layers. Each node and edge type gets its own parameters; not available on OpTC."),
    },
}

DECODERS_NODE_LEVEL = ["node_mlp", "none", "magic_gat", "nodlink"]
DECODERS_EDGE_LEVEL = ["edge_mlp"]
DECODERS_CFG = {
    "edge_mlp": {
        "architecture_str": Arg(
            str,
            desc="A string describing a simple neural network. Example: if the encoder's output has shape `node_out_dim=128` \
                                setting `architecture_str=linear(2) | relu | linear(0.5)` creates this MLP: nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 128), nn.Linear(128, y). \
                                Precisely, in linear(x), x is the multiplier of input neurons. The final layer `nn.Linear(128, y)` is added automatically such that `y` is the \
                                output size matching the downstream objective (e.g. edge type prediction involves predicting 10 edge types, so the output of the decoder should be 10).",
        ),
        "src_dst_projection_coef": Arg(
            int, desc="Multiplier of input neurons to project src and dst nodes."
        ),
    },
    "node_mlp": {
        "architecture_str": Arg(str),
    },
    "magic_gat": {
        "num_layers": Arg(int),
        "num_heads": Arg(int),
        "negative_slope": Arg(float),
        "alpha_l": Arg(float),
        "activation": Arg(str),
    },
    "nodlink": {},
    "inner_product": {},
    "none": {},
}

RECON_LOSSES = ["SCE", "MSE", "MSE_sum", "MAE", "none"]
PRED_LOSSES = ["cross_entropy", "BCE"]
OBJECTIVES_NODE_LEVEL = [
    "predict_node_type",
    "reconstruct_node_features",
    "reconstruct_node_embeddings",
    "reconstruct_masked_features",
    "one_class",
    "predict_masked_struct",
]
OBJECTIVES_EDGE_LEVEL = [
    "predict_edge_type",
    "predict_edge_supervised",
    "reconstruct_edge_embeddings",
    "predict_edge_contrastive",
]
OBJECTIVES = OBJECTIVES_NODE_LEVEL + OBJECTIVES_EDGE_LEVEL
OBJECTIVES_CFG = {
    # Prediction-based
    "predict_edge_supervised": {
        "decoder": Arg(
            str, vals=OR(list(DECODERS_CFG.keys())), desc="Decoder used before computing loss."
        ),
        **DECODERS_CFG,
        "mode": Arg(str, vals=OR(["scores", "patterns", "synthetic"]),
                    desc="'scores': pick top-N from an edge_scores pkl; 'patterns': match hand-crafted TTP patterns; "
                         "'synthetic': use pre-computed embeddings for explicit attack edge definitions."),
        # mode=scores fields
        "top_n_attacks": Arg(int, desc="(scores mode) Number of top-loss edges to use as attack examples."),
        # mode=patterns fields
        "attack_patterns": Arg(list, desc="(patterns mode) List of TTP pattern dicts."),
        "max_edges_per_pattern": Arg(int, desc="(patterns mode) Max edges collected per pattern."),
        # mode=synthetic fields
        "attack_edges_path": Arg(str, desc="(synthetic mode) Path to YAML file with explicit (src_type, src_label, "
                                           "edge_type, dst_type, dst_label) attack edge definitions."),
        "pos_weight": Arg(float, desc="BCE pos_weight for the attack class (on top of 1:1 oversampling)."),
    },
    "predict_edge_type": {
        "loss": Arg(str, vals=OR(PRED_LOSSES), desc="Loss used to predict the edge type."),
        "decoder": Arg(
            str, vals=OR(list(DECODERS_CFG.keys())), desc="Decoder used before computing loss."
        ),
        **DECODERS_CFG,
        "balanced_loss": Arg(bool),
        "use_triplet_types": Arg(bool),
        "AMS":
            {
                "version": Arg(int),
                "margin": Arg(float),
                "scale": Arg(int),
            },
    },
    "predict_node_type": {
        "decoder": Arg(
            str, vals=OR(list(DECODERS_CFG.keys())), desc="Decoder used before computing loss."
        ),
        **DECODERS_CFG,
        "balanced_loss": Arg(bool),
    },
    "predict_masked_struct": {
        "loss": Arg(str, vals=OR(PRED_LOSSES)),
        "decoder": Arg(
            str, vals=OR(list(DECODERS_CFG.keys())), desc="Decoder used before computing loss."
        ),
        **DECODERS_CFG,
        "balanced_loss": Arg(bool),
    },
    "detect_edge_few_shot": {
        "decoder": Arg(
            str, vals=OR(list(DECODERS_CFG.keys())), desc="Decoder used before computing loss."
        ),
        **DECODERS_CFG,
    },
    "predict_edge_contrastive": {
        "decoder": Arg(
            str, vals=OR(list(DECODERS_CFG.keys())), desc="Decoder used before computing loss."
        ),
        **DECODERS_CFG,
        "inner_product": {
            "dropout": Arg(float),
        },
    },
    # Reconstruction-based
    "reconstruct_node_features": {
        "loss": Arg(str, vals=OR(RECON_LOSSES)),
        "decoder": Arg(
            str, vals=OR(list(DECODERS_CFG.keys())), desc="Decoder used before computing loss."
        ),
        **DECODERS_CFG,
    },
    "reconstruct_node_embeddings": {
        "loss": Arg(str, vals=OR(RECON_LOSSES)),
        "decoder": Arg(
            str, vals=OR(list(DECODERS_CFG.keys())), desc="Decoder used before computing loss."
        ),
        **DECODERS_CFG,
    },
    "reconstruct_edge_embeddings": {
        "loss": Arg(str, vals=OR(RECON_LOSSES)),
        "decoder": Arg(
            str, vals=OR(list(DECODERS_CFG.keys())), desc="Decoder used before computing loss."
        ),
        **DECODERS_CFG,
    },
    "reconstruct_masked_features": {
        "loss": Arg(str, vals=OR(RECON_LOSSES)),
        "mask_rate": Arg(float),
        "decoder": Arg(
            str, vals=OR(list(DECODERS_CFG.keys())), desc="Decoder used before computing loss."
        ),
        **DECODERS_CFG,
    },
    "one_class": {
        "decoder": Arg(
            str, vals=OR(list(DECODERS_CFG.keys())),
            desc="Decoder applied to embeddings before the hypersphere; use 'none' (identity).",
        ),
        **DECODERS_CFG,
        "beta": Arg(float, desc="Soft-boundary fraction; radius is the (1-beta) distance quantile."),
        "eps": Arg(float, desc="Center slack keeping |c| away from zero."),
        "warmup": Arg(int, desc="Kept for parity; a no-op (c/r update every train step)."),
    },
}

SYNTHETIC_ATTACKS = {
    "synthetic_attack_naive": {
        "num_attacks": Arg(int),
        "num_malicious_process": Arg(int),
        "num_unauthorized_file_access": Arg(int),
        "process_selection_method": Arg(str),
    },
}

REQUIRE_HETERO_FEATURES_ENCODERS = ["hetero_graph_transformer"]
REQUIRE_NON_REVERSED_EDGES_ENCODERS = ["hetero_graph_transformer", "event_type_encoding"]

THRESHOLD_METHODS = ["max_val_loss", "mean_val_loss", "percentile", "threatrace", "magic", "flash", "nodlink", "fixed_zero", "ocrapt"]

# --- Tasks, subtasks, and argument configurations ---
TASK_ARGS = {
    "construction": {
        "used_method": Arg(
            str, vals=OR(["default", "magic"]), desc="The method to build time window graphs."
        ),
        "use_all_files": Arg(bool),
        "mimicry_edge_num": Arg(int),
        "time_window_size": Arg(
            float,
            desc="The size of each graph in minutes. The notation should always be float (e.g. 10.0). Supports sizes < 1.0.",
        ),
        "use_hashed_label": Arg(bool, desc="Whether to hash the textual features."),
        "fuse_edge": Arg(
            bool, desc="Whether to fuse duplicate sequential edges into a single edge."
        ),
        "consistent_edge_types": Arg(
            bool, desc="Map OpTC edge types to DARPA TC equivalents for shared vocabulary across datasets."
        ),
        "node_label_features": {
            "subject": Arg(
                str,
                vals=AND(["auto", "type", "path", "cmd_line"]),
                desc="Which features use for process nodes. Features will be concatenated. Use 'auto' to include type always and other attributes only when non-null.",
            ),
            "file": Arg(
                str,
                vals=AND(["auto", "type", "path"]),
                desc="Which features use for file nodes. Features will be concatenated. Use 'auto' to include type always and other attributes only when non-null.",
            ),
            "netflow": Arg(
                str,
                vals=AND(["auto", "type", "remote_ip", "remote_port"]),
                desc="Which features use for netflow nodes. Features will be concatenated. Use 'auto' to include type always and other attributes only when non-null.",
            ),
        },
        "null_label_tokens": Arg(
            bool,
            desc="When enabled, null entity attributes produce explicit tokens ([NO_CMD], [NO_PATH], [NO_IP]) instead of empty strings.",
        ),
        "multi_dataset": Arg(
            str,
            vals=OR(list(DATASET_DEFAULT_CONFIG.keys()) + ["none"]),
            desc="A comma-separated list of datasets on which training is performed. Evaluation is done only the primary dataset run in CLI.",
        ),
    },
    "transformation": {
        "used_methods": Arg(
            str,
            vals=AND(
                ["undirected", "dag", "rcaid_pseudo_graph", "none"]
                + list(SYNTHETIC_ATTACKS.keys())
            ),
            desc="Applies transformations to graphs after their construction. Multiple transformations can be applied sequentially. Example: `used_methods=undirected,dag`",
        ),
        "rcaid_pseudo_graph": {
            "use_pruning": Arg(bool),
        },
        **SYNTHETIC_ATTACKS,
    },
    "featurization": {
        "emb_dim": Arg(
            int,
            desc="Size of the text embedding. Arg not used by some featurization methods that do not build embeddings.",
        ),
        "epochs": Arg(
            int, desc="Epochs to train the embedding method. Arg not used by some methods."
        ),
        "seed": Arg(int),
        "training_split": Arg(
            str,
            vals=OR(["train", "all"]),
            desc="The partition of data used to train the featurization method.",
        ),
        "multi_dataset_training": Arg(
            bool,
            desc="Whether the featurization method should be trained on all datasets in `multi_dataset`.",
        ),
        "pretrain_datasets": Arg(
            str,
            desc="Comma-separated list of dataset names for multi-dataset pretraining (used by word2vec, doc2vec, etc.).",
        ),
        "used_method": Arg(
            str,
            vals=OR(list(FEATURIZATIONS_CFG.keys())),
            desc="Algorithms used to create node and edge features.",
        ),
        **FEATURIZATIONS_CFG,
    },
    "feat_inference": {
        "to_remove": Arg(bool),  # TODO: remove
        "continue_pretrain": Arg(bool, desc="Continue pretraining on the target dataset before inference. Default: False."),
        "continue_pretrain_epochs": Arg(int, desc="Number of epochs for continue-pretraining on target dataset. Default: 3."),
        "continue_pretrain_lr_factor": Arg(float, desc="LR multiplier relative to original peak LR (e.g., 0.1 = 10x lower). Default: 0.1."),
        "edge_batch_size": Arg(int, desc="Number of edges to process in each batch during feat_inference. Default: 5000."),
        "rename_attack": {
            "enabled": Arg(bool, desc="If True, after embeddings are computed, replace the embedding of selected test-set nodes with a benign target embedding. Simulates an inference-time renaming/mimicry attack."),
            "entities": Arg(str, desc="Comma-separated BASE NAMES of attack entities (e.g. 'main, XIM, sendmail'). Each is matched two ways against test-set nodes of `target_node_type`: (a) exact label equality, and (b) labels ending with '/<base>' (path-style)."),
            "target_node_type": Arg(str, vals=OR(["subject", "file", "netflow"]), desc="Node type the attack targets (only nodes of this type are rewritten)."),
            "top_k": Arg(int, desc="If >0, automatically build a benign-target pool from the top-K most frequent BARE labels (no '/') of `target_node_type` seen in the training split, and assign one target per attack entity. Path-suffix matches preserve the original prefix (e.g. '/tmp/main' -> '/tmp/<assigned-target>'). If 0, use the manual `target_label` for every entity (e.g. for an 'unseen-by-pretraining' target)."),
            "target_label": Arg(str, desc="(Manual mode only, used when top_k=0) Bare base name to replace every attack entity. Path-suffix matches still preserve the prefix (e.g. '/tmp/main' -> '/tmp/<target_label>')."),
            "seed": Arg(int, desc="Seed used to shuffle the auto-picked benign pool before assigning one target per entity. Vary it for sensitivity analysis over which benign label each malicious entity is mimicking."),
        },
    },
    "batching": {
        "save_on_disk": Arg(
            bool,
            desc="Whether to store the graphs on disk upon building the graphs. \
            Used to avoid re-computation of very complex batching operations that take time. Can take up to 300GB storage for CADETS_E5.",
        ),
        "node_features": Arg(
            str,
            vals=AND(["node_type", "node_emb", "only_ones", "edges_distribution"]),
            desc="Node features to use during GNN training. `node_type` is a one-hot encoded entity type vector, \
                                    `node_emb` refers to the embedding generated during the `featurization` task, `only_ones` is a vector of ones \
                                    with length `node_type`, `edges_distribution` counts emitted and received edges.",
        ),
        "edge_features": Arg(
            str,
            vals=AND(["edge_type", "edge_type_triplet", "msg", "time_encoding", "none"]),
            desc="Edge features to used during GNN training. `edge_type` refers to the system call type, `edge_type_triplet` \
                                considers a same edge type as a new type if source or destination node types are different, `msg` is the message vector \
                                used in the TGN, `time_encoding` encodes temporal order of events with their timestamps in the TGN, `none` uses no features.",
        ),
        "multi_dataset_training": Arg(
            bool, desc="Whether the GNN should be trained on all datasets in `multi_dataset`."
        ),
        "fix_buggy_graph_reindexer": Arg(
            bool,
            desc="A bug has been found in the first version of the framework, where reindexing graphs in shape (N, d) \
                                                slightly modify node features. Setting this to true fixes the bug.",
        ),
        "global_batching": {
            "used_method": Arg(
                str,
                vals=OR(["edges", "minutes", "unique_edge_types", "none"]),
                desc="Flattens the time window-based graphs into a single large \
                            temporal graph and recreate graphs based on the given method. `edges` creates contiguous graphs of size `global_batching_batch_size` edges, \
                            the same applies for `minutes`, `unique_edge_types` builds graphs where each pair of connected nodes share edges with distinct edge types, \
                            `none` uses the default time window-based batching defined in minutes with arg `time_window_size`.",
            ),
            "global_batching_batch_size": Arg(
                int,
                desc="Controls the value associated with `global_batching.used_method` (training+inference).",
            ),
            "global_batching_batch_size_inference": Arg(
                int,
                desc="Controls the value associated with `global_batching.used_method` (inference only).",
            ),
        },
        "intra_graph_batching": {
            "used_methods": Arg(
                str,
                vals=AND(["edges", "tgn_last_neighbor", "none"]),
                desc="Breaks each previously computed graph into even smaller graphs. \
                                `edges` creates contiguous graphs of size `intra_graph_batch_size` edges (if a graph has 2000 edges and `intra_graph_batch_size=1500` \
                                creates two graphs: one with 1500 edges, the other with 500 edges), `tgn_last_neighbor` computes for each graph its associated graph \
                                based on the TGN last neighbor loader, namely a new graph where each node is connected with its last `tgn_neighbor_size` incoming edges.\
                                `none` does not alter any graph.",
            ),
            "edges": {
                "intra_graph_batch_size": Arg(
                    int,
                    desc="Controls the value associated with `global_batching.used_method`.",
                ),
            },
            "tgn_last_neighbor": {
                "tgn_neighbor_size": Arg(
                    int, desc="Number of last neighbors to store for each node."
                ),
                "tgn_neighbor_n_hop": Arg(
                    int,
                    desc="If greater than one, will also gather the last neighbors of neighbors.",
                ),
                "fix_buggy_orthrus_TGN": Arg(
                    bool,
                    desc="A bug has been in the first version of the framework, where the features of last neighbors not appearing \
                                            in the input graph have zero node feature vectors. Setting this arg to true includes the features of all nodes in the TGN graph.",
                ),
                "fix_tgn_neighbor_loader": Arg(
                    bool,
                    desc="We found a minor bug in the original TGN code (https://github.com/pyg-team/pytorch_geometric/issues/10100). This \
                                                is an unofficial fix.",
                ),
                "directed": Arg(
                    bool,
                    desc="The original TGN's loader builds graphs in an undirected way. This makes the graphs purely directed.",
                ),
                "insert_neighbors_before": Arg(
                    bool,
                    desc="Whether to insert the edges of the current graph before loading last neighbors.",
                ),
            },
        },
        "inter_graph_batching": {
            "used_method": Arg(
                str,
                vals=OR(["graph_batching", "none"]),
                desc="Batches multiple graphs into a single large one for parallel training. \
                                Does not support TGN. `graph_batching` batches `inter_graph_batch_size` together, `none` doesn't batch graphs.",
            ),
            "inter_graph_batch_size": Arg(
                int,
                desc="Controls the value associated with `inter_graph_batching.used_method`.",
            ),
        },
    },
    "training": {
        "seed": Arg(int),
        "deterministic": Arg(bool, desc="Whether to force PyTorch to use deterministic algorithms."),
        "num_epochs": Arg(int),
        "patience": Arg(int),
        "lr": Arg(float),
        "weight_decay": Arg(float),
        "node_hid_dim": Arg(int, desc="Number of neurons in the middle layers of the encoder."),
        "node_out_dim": Arg(int, desc="Number of neurons in the last layer of the encoder."),
        "grad_accumulation": Arg(int, desc="Number of epochs to gather gradients before backprop."),
        "stable_optim": Arg(
            bool, desc="Use AdamW + warmup cosine schedule + gradient clipping for stable training."
        ),
        "inference_device": Arg(str, vals=OR(["cpu", "cuda"]), desc="Device used during testing."),
        "fuse_duplicate_edges_training": Arg(bool, desc="During training only, keeps one edge per (source, destination) pair in each batch, so that repeated events between the same entities count once in the loss."),
        "used_method": Arg(str, vals=OR(["default", "spider"]), desc="Which training pipeline use."),
        "ocrapt_early_stop": {
            "enabled": Arg(bool, desc="Off by default, no-op unless set."),
            "patience": Arg(int),
            "min_delta": Arg(float),
            "max_delta": Arg(float),
        },
        "encoder": {
            "dropout": Arg(float),
            "used_methods": Arg(
                str,
                vals=AND(list(ENCODERS_CFG.keys())),
                desc="First part of the neural network. Usually GNN encoders to capture complex patterns.",
            ),
            "x_is_tuple": Arg(
                bool, desc="Whether to consider nodes differently when being source or destination."
            ),
            **ENCODERS_CFG,
        },
        "decoder": {
            "used_methods": Arg(
                str,
                vals=AND(list(OBJECTIVES_CFG.keys())),
                desc="Second part of the neural network. Usually MLPs specific to the downstream task (e.g. reconstruction of prediction)",
            ),
            **OBJECTIVES_CFG,
            "use_few_shot": Arg(bool, desc="Old feature: need some work to update it."),
            "few_shot": {
                "include_attacks_in_ssl_training": Arg(bool),
                "freeze_encoder": Arg(bool),
                "num_epochs_few_shot": Arg(int),
                "patience_few_shot": Arg(int),
                "lr_few_shot": Arg(float),
                "weight_decay_few_shot": Arg(float),
                "decoder": {
                    "used_methods": Arg(str),
                    **OBJECTIVES_CFG,
                },
            },
        },
        "spider": {
            "finetune_mode": Arg(str, desc="Fine-tuning and scoring mode: 'cls', 'cls_attack', 'lp', 'mlm', 'tgn', 'edge_cls', or 'perplexity'."),
            "finetune_epochs": Arg(int, desc="Number of fine-tuning epochs."),
            "finetune_walk_len": Arg(int, desc="Number of nodes per context walk during fine-tuning and inference."),
            "finetune_lr": Arg(float, desc="Learning rate for fine-tuning."),
            "finetune_margin": Arg(float, desc="Margin for the ranking loss in cls mode. The anomalous logit must exceed the normal logit by at least this value. Larger values spread anomaly scores further apart."),
            "freeze_backbone": Arg(bool, desc="Freeze pretrained backbone weights during fine-tuning (only train the head)."),
            "num_inference_walks": Arg(int, desc="Number of context walks per edge at inference. Scores are averaged to reduce variance."),
            "inference_batch_size": Arg(int, desc="Number of edges per batch during inference scoring."),
            "edge_score_weight": Arg(float, desc="Weight (lambda) for edge-type CE in combined scoring: score = node_CE + lambda * edge_CE. Requires mask_edge_type=True."),
            "num_attack_walks": Arg(int, desc="Number of unique context walks to extract per malicious edge for cls_attack fine-tuning. Walks are deduplicated by edge-type sequence."),
            "event_emb": {
                "pool_method": Arg(str, vals=OR(["mean"]), desc="How to extract embeddings from BERT hidden states over a node span. 'mean' = mean-pool."),
            },
            "tgn": {
                "objective": Arg(str, desc="TGN training objective: 'contrastive' (BCE with neg sampling), 'edge_pred' (predict edge type), or 'node_pred' (predict dst node type)."),
                "memory_dim": Arg(int, desc="Entity memory dimension for TGN mode (default: same as BERT hidden size)."),
                "edge_emb_dim": Arg(int, desc="Edge type embedding dimension for TGN mode."),
                "time_dim": Arg(int, desc="Time encoding dimension for TGN mode."),
                "num_heads": Arg(int, desc="Number of attention heads in TGN memory updater."),
                "batch_size": Arg(int, desc="Temporal batch size (edges per update step) for TGN mode."),
                "use_node_type_emb": Arg(bool, desc="Include src/dst node type embeddings in the TGN classifier input."),
                "use_time_emb": Arg(bool, desc="Include src/dst time-delta encodings in the TGN classifier input."),
                "use_memory": Arg(bool, desc="Include evolving memory vectors in the edge predictor/classifier input. When False, predictions use only static BERT embeddings."),
                "use_entity_emb": Arg(bool, desc="Include static BERT entity embeddings in the edge predictor/classifier input. When False, predictions rely on memory and other features."),
                "use_event_bert_emb": Arg(bool, desc="(event mode only) Include per-edge contextual src/dst BERT embeddings in TGN prediction heads."),
                "reset_memory_on_inference": Arg(bool, desc="Start inference with empty memory instead of trained memory state. Simulates production deployment on unseen data."),
                "score_gated_memory": Arg(bool, desc="Gate messages by per-edge anomaly score before aggregation. High-anomaly edges leave stronger memory traces; low-anomaly edges are suppressed."),
                "temporal_decay": Arg(bool, desc="Apply exponential memory decay toward zero based on time elapsed since last update. Learned decay rate. Old events naturally fade."),
                "anomaly_accumulator": Arg(bool, desc="Track per-node EMA of anomaly scores. When temporal_decay is also True, nodes with high accumulated anomaly decay slower (stickier memory)."),
            },
        },
    },
    "evaluation": {
        "viz_malicious_nodes": Arg(
            bool,
            desc="Whether to generate images of malicious nodes' neighborhoods (not stable).",
        ),
        "ground_truth_version": Arg(str, vals=OR(["orthrus", "reapr", "threatrace"])),
        "best_model_selection": Arg(
            str,
            vals=OR(["best_adp", "best_discrimination", "best_ap@10"]),
            desc="Strategy to select the best model across epochs. `best_adp` selects the best model based on the highest ADP score, `best_discrimination` \
                                    selects the model that does the best separation between top-score TPs and top-score FPs.",
        ),
        "used_method": Arg(str),
        "node_evaluation": {
            "threshold_method": Arg(
                str,
                vals=OR(THRESHOLD_METHODS),
                desc="Method to calculate the threshold value used to detect anomalies.",
            ),
            "max_val_loss": {
                "alpha": Arg(float, desc="Weights the margin applied to the threshold value."),
            },
            "use_dst_node_loss": Arg(
                bool,
                desc="Whether to consider the loss of destination nodes when computing the node-level scores (maximum loss of a node).",
            ),
            "use_kmeans": Arg(
                bool, desc="Whether to cluster nodes after thresholding as done in Orthrus"
            ),
            "kmeans_top_K": Arg(int, desc="Number of top-score nodes selected before clustering."),
            "ocrapt_contamination": Arg(
                float,
                desc="For threshold_method=ocrapt: max per-type contamination (top fraction "
                "flagged), clamped from that type's own val malicious fraction.",
            ),
            "ocrapt_min_contamination": Arg(
                float,
                desc="For threshold_method=ocrapt: min per-type contamination floor.",
            ),
        },
        "tw_evaluation": {
            "threshold_method": Arg(
                str,
                vals=OR(THRESHOLD_METHODS),
                desc="Time-window detection. The code is broken and needs work to be updated.",
            ),
        },
        "node_tw_evaluation": {
            "threshold_method": Arg(
                str,
                vals=OR(THRESHOLD_METHODS),
                desc="Node-level detection where a same node in multiple time windows is \
                    considered as multiple unique nodes. More realistic evaluation for near real-time detection. The code is broken and needs work to be updated.",
            ),
            "use_dst_node_loss": Arg(bool),
            "use_kmeans": Arg(bool),
            "kmeans_top_K": Arg(int),
        },
        "queue_evaluation": {
            "used_method": Arg(
                str,
                vals=OR(["kairos_idf_queue", "provnet_lof_queue"]),
                desc="Queue-level detection as in Kairos. The code is broken and needs work to be updated.",
            ),
            "queue_threshold": Arg(int),
            "kairos_idf_queue": {
                "include_test_set_in_IDF": Arg(bool),
            },
            "provnet_lof_queue": {
                "queue_arg": Arg(str),
            },
        },
        "edge_evaluation": {
            "malicious_edge_selection": Arg(
                str,
                vals=OR(["src_node", "dst_node", "both_nodes"]),
                desc="The ground truth only contains node-level labels. \
                This arg controls the strategy to label edges. `src_nodes` and `dst_nodes` consider an edge as malicious if only its source or only its destination \
                node is malicious. `both` labels an edge as malicious if both end nodes are malicious.",
            ),
            "threshold_method": Arg(str, vals=OR(THRESHOLD_METHODS)),
        },
    },
    "triage": {
        "used_method": Arg(
            str,
            vals=OR(["depimpact", "ocrapt_subgraph"]),
            desc="Post-processing step to reconstruct attack paths or reduce false positives. `depimpact` is used in Orthrus; `ocrapt_subgraph` is OCR-APT's anomalous-subgraph stage.",
        ),
        "depimpact": {
            "used_method": Arg(
                str, vals=OR(["component", "shortest_path", "1-hop", "2-hop", "3-hop"])
            ),
            "score_method": Arg(str, vals=OR(["degree", "recon_loss", "degree_recon"])),
            "workers": Arg(int),
            "visualize": Arg(bool),
        },
        "ocrapt": {
            "num_hops": Arg(int, desc="hops for correlating anomalies into subgraphs"),
            "top_k": Arg(int, desc="top-K seed nodes per node type (by Anomaly_score)"),
            "min_nodes": Arg(int, desc="minimum nodes per constructed subgraph"),
            "max_edges": Arg(int, desc="subgraphs above this are Louvain-partitioned + edge-sampled"),
            "abnormality_level": Arg(
                str, vals=OR(["Negligible", "Minor", "Moderate", "Significant", "Critical"]),
                desc="least subgraph severity to keep (summed Prediction_probability)",
            ),
            "correlate_anomalous_once": Arg(bool),
            "remove_duplicated_subgraph": Arg(bool),
        },
    },
    "postprocessing": {},
}

EXPERIMENTS_CONFIG = {
    "training_loop": {
        "run_evaluation": Arg(
            str, vals=OR(["each_epoch", "best_epoch"])
        ),  # (when to run inference on test set)
    },
    "experiment": {
        "used_method": Arg(str, vals=OR(["uncertainty", "none"])),
        "uncertainty": {
            "hyperparameter": {
                "hyperparameters": Arg(str, vals=AND(["lr, num_epochs, text_h_dim, gnn_h_dim"])),
                "iterations": Arg(int),
                "delta": Arg(float),
            },
            "mc_dropout": {
                "iterations": Arg(int),
                "dropout": Arg(float),
            },
            "deep_ensemble": {
                "method": Arg(str),
                "iterations": Arg(int),
                "restart_from": Arg(str),
            },
            "bagged_ensemble": {
                "iterations": Arg(int),
                "min_num_days": Arg(int),
            },
        },
    },
}
UNCERTAINTY_EXP_YML_FOLDER = "experiments/uncertainty/"
