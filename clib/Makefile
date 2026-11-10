# SPDX-License-Identifier: Apache-2.0
# libreflop: the reFLOP C/C++ API.
#   make            build build/libreflop.so and the examples
#   make check      run the examples against your default paired server
ROOT := ..
BUILD := build
CFLAGS ?= -O2 -Wall -Wextra -Iinclude
CXXFLAGS ?= -O2 -Wall -Wextra -std=c++17 -Iinclude
LDFLAGS += -L$(BUILD) -Wl,-rpath,'$$ORIGIN' -lreflop -lm

all: $(BUILD)/libreflop.so $(BUILD)/matmul $(BUILD)/mlp_forward

$(BUILD)/libreflop.so: $(wildcard go/*.go) include/reflop.h
	@mkdir -p $(BUILD)
	cd $(ROOT) && CGO_ENABLED=1 go build -trimpath -buildmode=c-shared -o clib/$(BUILD)/libreflop.so ./clib/go
	rm -f $(BUILD)/libreflop.h  # the public header is include/reflop.h

$(BUILD)/matmul: examples/matmul.c $(BUILD)/libreflop.so
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(BUILD)/mlp_forward: examples/mlp_forward.cpp include/reflop.hpp $(BUILD)/libreflop.so
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

check: all
	$(BUILD)/matmul
	$(BUILD)/mlp_forward

clean:
	rm -rf $(BUILD)

.PHONY: all check clean
