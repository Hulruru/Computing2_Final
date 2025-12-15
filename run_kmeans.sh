# 다음과 같이 권한 주고 실행
# chmod +x run_all_kmeans.sh
# ./run_all_kmeans.sh

#!/bin/bash

set -e

# ================= 설정 =================
NVCC=nvcc

SRC1=mNaiveKmeans.cu
SRC2=mSoAKmeans.cu
SRC3=mCSRKmeans.cu

EXE1=mNaiveKmeans
EXE2=mSoAKmeans
EXE3=mCSRKmeans

RESULT_DIR=result

PYTHON_SCRIPT="cuml_KMeans.py"
BATCH_SIZE=5
TOTAL_BATCHES=6
# ========================================

mkdir -p ${RESULT_DIR}
# ================= sklearn K-means =================
python3 "sklearn_KMeans.py" ${CUMUL_RESULT_FILE} 2>&1
# ================= CUDA K-means =================
${NVCC} ${SRC1} -o ${EXE1} -O3 -arch=sm_86
./${EXE1} > ${RESULT_DIR}/naive_result.txt

${NVCC} ${SRC2} -o ${EXE2} -O3 -arch=sm_86
./${EXE2} > ${RESULT_DIR}/soa_result.txt

${NVCC} ${SRC3} -o ${EXE3} -O3 -arch=sm_86
./${EXE3} > ${RESULT_DIR}/csr_result.txt

# ================= cuML K-means =================
CUMUL_RESULT_FILE=${RESULT_DIR}/cuml_result.txt
: > ${CUMUL_RESULT_FILE}   # 파일 초기화

for i in $(seq 0 $((TOTAL_BATCHES - 1))); do
    START_INDEX=$((i * BATCH_SIZE))
    END_INDEX=$((START_INDEX + BATCH_SIZE - 1))

    python3 "${PYTHON_SCRIPT}" "${START_INDEX}" "${END_INDEX}" \
        >> ${CUMUL_RESULT_FILE} 2>&1
done
