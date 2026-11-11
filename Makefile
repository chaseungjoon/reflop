# SPDX-License-Identifier: Apache-2.0
VERSION ?= $(shell git describe --tags --always --dirty 2>/dev/null || echo dev)
COMMIT  ?= $(shell git rev-parse HEAD 2>/dev/null)
DATE    ?= $(shell date -u +%Y-%m-%dT%H:%M:%SZ)
PKG     := github.com/chaseungjoon/reflop/server/reflopd/internal/buildinfo
LDFLAGS := -s -w -X $(PKG).Version=$(VERSION) -X $(PKG).Commit=$(COMMIT) -X $(PKG).Date=$(DATE)
PYTHON  ?= $(CURDIR)/.venv/bin/python

.PHONY: all build build-probes clib wheel test test-go test-python test-isolation lint codegen codegen-check fmt clean sign-release

all: build

## build: bin/reflopd (static, embeds the worker)
build:
	CGO_ENABLED=0 go build -trimpath -ldflags "$(LDFLAGS)" -o bin/reflopd ./server/reflopd/cmd/reflopd

## build-probes: bin/reflopd-probes, a TEST build with isolation probes (never ship it)
build-probes:
	CGO_ENABLED=0 go build -tags reflop_probes -trimpath -ldflags "$(LDFLAGS)" -o bin/reflopd-probes ./server/reflopd/cmd/reflopd

## clib: clib/build/libreflop.so and the C/C++ examples
clib:
	$(MAKE) -C clib

## wheel: dist/reflop-<version>-py3-none-any.whl (the Python client)
wheel:
	cd client/python && $(PYTHON) -m pip wheel --no-deps -w ../../dist .

test: test-go test-python

test-go:
	go test -race ./...

## test-python: worker validation/execution, client capture, security corpus, conformance
test-python:
	$(PYTHON) -m pytest -q tests

## test-isolation: sandbox probes inside real workers (needs an installed runtime)
test-isolation:
	REFLOP_TEST_HOME=$${REFLOP_TEST_HOME:?set REFLOP_TEST_HOME to a REFLOP_HOME with a runtime} \
		go test -count=1 -tags reflop_probes ./server/reflopd/internal/server -run Isolation -v

lint:
	test -z "$$(gofmt -l server pkg clib)"
	go vet ./...
	$(PYTHON) -m ruff check client/python server/worker tests codegen spikes
	python3 tests/security/static/check_worker.py

## codegen: regenerate protobuf and opset code (needs protoc 29.x and protoc-gen-go)
codegen:
	codegen/gen_proto.sh
	$(PYTHON) codegen/gen.py

codegen-check:
	$(PYTHON) codegen/gen.py --check

fmt:
	gofmt -w server pkg clib
	$(PYTHON) -m ruff check --fix client/python server/worker tests codegen spikes

## sign-release: sign and publish the draft release TAG with the offline key (scripts/sign-release.sh)
sign-release:
	scripts/sign-release.sh $${TAG:?set TAG=vX.Y.Z}

clean:
	rm -rf bin dist clib/build
