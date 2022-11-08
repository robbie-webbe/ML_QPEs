#!/bin/bash/

wget -O source_lc.tar "https://nxsa.esac.esa.int/nxsa-sl/servlet/data-action-aio?obsno=$1&sourceno=$2&level=PPS&instname=PN&extension=FTZ&name=SRCTSR"

tar -xvf source_lc.tar
rm source_lc.tar
