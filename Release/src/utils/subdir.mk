################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/utils/read_graph.cpp 

OBJS += \
./src/utils/read_graph.o 

CPP_DEPS += \
./src/utils/read_graph.d 


# Each subdirectory must supply rules for building sources it contributes
src/utils/%.o: ../src/utils/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	nvcc -O3 -gencode arch=compute_35,code=sm_35  -odir "src/utils" -M -o "$(@:%.o=%.d)" "$<"
	nvcc -O3 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


