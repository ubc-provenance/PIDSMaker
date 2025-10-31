## Downloading Files Manually

1. create a new folder (referred to as the *DATA_FOLDER*) and download all `.gz` files from a specific DARPA dataset (follow the link provided for DARPA E3 [here](https://drive.google.com/drive/folders/1fOCY3ERsEmXmvDekG-LUUSjfWs6TRdp-) and DARPA E5 [here](https://drive.google.com/drive/folders/1GVlHQwjJte3yz0n1a1y4H4TfSe8cu6WJ)). If using CLI, [use gdown](https://stackoverflow.com/a/50670037/10183259), by taking the ID of the document directly from the URL. In some cases, the downloading of a file may stop, in this case, simply ctrl+C and re-run the same gdown command with `--continue` until the file is fully downloaded. 
**NOTE:** Old files should be deleted before  downloading a new dataset.

2. in the DATA_FOLDER, download the java binary (ta3-java-consumer.tar.gz) to build the avro files for DARPA [E3](https://drive.google.com/drive/folders/1kCRC5CPI8MvTKQFvPO4hWIRHeuUXLrr1) and [E5](https://drive.google.com/drive/folders/1YDxodpEmwu4VTlczsrLGkZMnh_o70lUh).

3. in the DATA_FOLDER, download the schema files (i.e. files with filename extension '.avdl' and '.avsc') for DARPA [E3](https://drive.google.com/drive/folders/1gwm2gAlKHQnFvETgPA8kJXLLm3L-Z3H1) and [E5](https://drive.google.com/drive/folders/1fCdYCIMBCm7gmBpmqDTuoMhbOoIY6Wzq).
