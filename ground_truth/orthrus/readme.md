# Ground Truth darpa_v4

## DARPA Official Documents
- E5: [official document pdf of E5](../TA51_Final_report_E5.pdf)
- E3: [official document pdf of E3](../TC_Ground_Truth_Report_E3_Update.pdf)

## Theia E3
| attack                                   | Date | Description     | TPs | Status  |
|------------------------------------------|------|-----------------|-----|---------|
| Firefox_Backdoor_Drakon_In_Memory        | 0410 | succeed         | 58  | used    |
| Browser_Extension_Drakon_Dropper         | 0412 | partial succeed | 61  | used    |
| Phishing E-mail w/ Link                  | 0410 | non-host        | -   | removed |
| Phishing E-mail w/ Executable Attachment | 0413 | fail            | -   | removed |

- **Browser_Extension_Drakon_Dropper** was partially successful, where `drakon loading` failed but 
`Micro apt` exploiting succeeded. We keep it in usage because it contains all activities of `Micro apt`.

## Cadets E3
| attack            | Date | Description     | TPs | Status  |
|-------------------|------|-----------------|-----|---------|
| Nginx_Backdoor_06 | 0406 | partial succeed | 8   | used    |
| Nginx_Backdoor_11 | 0406 | partial succeed | -   | removed |
| Nginx_Backdoor_12 | 0406 | partial succeed | 43  | used    |
| Nginx_Backdoor_13 | 0406 | partial succeed | 24  | used    |
| E_mail_Server     | 0406 | non-host        | -   | removed |

- **Nginx Backdoor attacks** are all partially succeeded (or half failed). 
  - On 6th, succeeded in `loaderDrakon connect to shell`, `download and run netrecon`,
  `inject libdrakon`. But failed in `inject sshd PID 809`.
  - On 11th, succeeded in `run drakon in memory`. But
  failed in `download libdrakon implant.so and inject sshd`.
  - On 12th, succeeded in `run drakon in memory`, `download drakon implant and elevate 
  privilege` and `run micro without root`. But failed in `elevate micro apt`.
  - On 13th, succeeded in `get shell` and `download files`. But failed in `elevate micro apt
  with new module`.

- **E_mail_Server** sent phishing emails, no host activities.

DONE: remove Nginx_Backdoor_11

## Trace E3
| attack                            | Date | Description     | TPs | Status  |
|-----------------------------------|------|-----------------|-----|---------|
| trace_e3_firefox_0410             | 0410 | succeed         | 11  | used    |
| trace_e3_phishing_executable_0413 | 0413 | partial succeed | 11  | used    |
| trace_e3_pine_0413                | 0413 | succeed         | 14  | used    |
| trace_e3_browser_extension_0412   | 0412 | failed          | -   | removed |
| trace_e3_phishing_link_0410       | 0410 | non-host        | -   | removed |

- **trace_e3_phishing_executable_0413** failed to `exploit pine vulnerability` and `open shell`. But 
succeed in `run micro apt and port scan` because host user opened and ran malicious executable attached
with the phishing email.

## Fivedirections E3
| attack                             | Date | Description | TPs | Status  |
|------------------------------------|------|-------------|-----|---------|
| fivedirections_e3_excel_0409       | 0409 | failed      | 63  | used    |
| fivedirections_e3_firefox_0411     | 0411 | succeed     | 56  | used    |
| fivedirections_e3_browser_0412     | 0412 | failed      | -   | removed |
| fivedirections_e3_executable_0413  | 0413 | failed      | -   | removed |

- **fivedirections_e3_excel_0409** malicious shell codes are inserted to a excel and failed to run 
automately. However, user manually copy the malicious command to a shell and ran it successfully. 
Should I consider it as a successful attack or a failed attack and some normal user activities?

DONE: remove fivedirections_e3_browser_0412
DONE: add fivedirections_e3_excel_0409

# Theia E5
| attack                                           | Date | Description | TPs | Status  |
|--------------------------------------------------|------|-------------|-----|---------|
| THEIA_1_Firefox_Drakon_APT_BinFmt_Elevate_Inject | 0515 | succeed     | 70  | used    |
| Firefox_Drakon_APT                               | 0514 | fail        | -   | removed |

# Cadets E5
| attack               | Date | Description | TPs | Status |
|----------------------|------|-------------|-----|--------|
| Nginx_Drakon_APT     | 0516 | succeed     | 19  | used   |
| Nginx_Drakon_APT_17  | 0517 | succeed     | 107 | used   |

In both two attacks, attackers tried twice and succeeded in the second try.

## Trace E5
| attack               | Date | Description | TPs | Status   |
|----------------------|------|-------------|-----|----------|
| Trace_Firefox_Drakon | 0514 | succeed     | 71  | used     |
| Azazel APT (Failed)  | 0517 | fail        | -   | removed  |

## Fivedirections E5
| attack                          | Date | Description     | TPs | Status |
|---------------------------------|------|-----------------|-----|--------|
| fivedirections_e5_copykatz_0509 | 0509 | partial succeed | 87  | used   |
| fivedirections_e5_bits_0515     | 0515 | succeed         | 54  | used   |
| fivedirections_e5_dns_0517      | 0517 | succeed         | 11  | used   |
| fivedirections_e5_drakon_0517   | 0517 | succeed         | 6   | used   |

- **fivedirections_e5_copykatz_0509** succeeded in `expolit firefox` and `install and run drakon`. 
But failed in `WMI` due to config issue.