################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../src/Lab1.cu 

OBJS += \
./src/Lab1.o 

CU_DEPS += \
./src/Lab1.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-10.1/bin/nvcc -G -g -O0 -gencode arch=compute_61,code=sm_61  -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-10.1/bin/nvcc -G -g -O0 --compile --relocatable-device-code=false -gencode arch=compute_61,code=compute_61 -gencode arch=compute_61,code=sm_61  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

