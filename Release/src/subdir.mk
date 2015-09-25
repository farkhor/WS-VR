################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../src/cuda_kernels.cu \
../src/entry.cu \
../src/iterative_procedure.cu \
../src/prepare_and_process_graph.cu 

CU_DEPS += \
./src/cuda_kernels.d \
./src/entry.d \
./src/iterative_procedure.d \
./src/prepare_and_process_graph.d 

OBJS += \
./src/cuda_kernels.o \
./src/entry.o \
./src/iterative_procedure.o \
./src/prepare_and_process_graph.o 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	nvcc -O3 -gencode arch=compute_35,code=sm_35  -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	nvcc -O3 --compile --relocatable-device-code=false -gencode arch=compute_35,code=compute_35 -gencode arch=compute_35,code=sm_35  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


