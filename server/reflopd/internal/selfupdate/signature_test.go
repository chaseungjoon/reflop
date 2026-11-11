// SPDX-License-Identifier: Apache-2.0

package selfupdate

import (
	"bytes"
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
)

// testdata holds a signature made by a throwaway key (test-signer.pub,
// "reflop-test-only"), never by a release key:
//
//	ssh-keygen -Y sign -f testkey -n reflop-release checksums.txt
func readTestdata(t *testing.T, name string) []byte {
	t.Helper()
	b, err := os.ReadFile("testdata/" + name)
	if err != nil {
		t.Fatal(err)
	}
	return b
}

func testKeys(t *testing.T) []string {
	return []string{strings.TrimSpace(string(readTestdata(t, "test-signer.pub")))}
}

// SEC_SWEEP M4: checksums.txt is trusted only with a valid signature by a
// release key, in the reflop-release namespace, over exactly these bytes.
func TestVerifySignature(t *testing.T) {
	msg, sig := readTestdata(t, "checksums.txt"), readTestdata(t, "checksums.txt.sig")
	if err := VerifySignature(msg, sig, testKeys(t)); err != nil {
		t.Fatalf("valid signature rejected: %v", err)
	}
	cases := map[string]func() error{
		"tampered message": func() error {
			return VerifySignature(bytes.Replace(msg, []byte("abc123"), []byte("abc124"), 1), sig, testKeys(t))
		},
		"untrusted key":   func() error { return VerifySignature(msg, sig, ReleaseKeys) },
		"wrong namespace": func() error { return VerifySignature(msg, readTestdata(t, "wrong-namespace.sig"), testKeys(t)) },
		"not a signature": func() error { return VerifySignature(msg, []byte("abc123  reflopd.tar.gz\n"), testKeys(t)) },
		"empty":           func() error { return VerifySignature(msg, nil, testKeys(t)) },
		"truncated": func() error {
			return VerifySignature(msg, append(sig[:len(sig)/2:len(sig)/2], []byte("\n-----END SSH SIGNATURE-----\n")...), testKeys(t))
		},
	}
	for name, f := range cases {
		if err := f(); !errors.Is(err, ErrSignature) {
			t.Errorf("%s: got %v, want ErrSignature", name, err)
		}
	}
}

// Fetch refuses a release whose signature is missing or does not verify,
// before it downloads the archive.
func TestFetchRequiresSignature(t *testing.T) {
	msg, sig := readTestdata(t, "checksums.txt"), readTestdata(t, "checksums.txt.sig")
	for _, tc := range []struct {
		name string
		sig  []byte // nil: no .sig asset
	}{{"missing", nil}, {"untrusted", sig}} {
		archiveFetched := false
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			switch {
			case strings.HasSuffix(r.URL.Path, "/checksums.txt"):
				w.Write(msg)
			case strings.HasSuffix(r.URL.Path, "/checksums.txt.sig") && tc.sig != nil:
				w.Write(tc.sig)
			case strings.HasSuffix(r.URL.Path, ".tar.gz"):
				archiveFetched = true
				w.Write([]byte("not reached"))
			default:
				http.NotFound(w, r)
			}
		}))
		u := &Updater{BaseURL: srv.URL, OS: "linux", Arch: "amd64", Client: srv.Client(), Keys: ReleaseKeys}
		_, err := u.Fetch(context.Background(), "v9.9.9")
		srv.Close()
		if !errors.Is(err, ErrSignature) {
			t.Errorf("%s: got %v, want ErrSignature", tc.name, err)
		}
		if archiveFetched {
			t.Errorf("%s: the archive was downloaded before the signature was checked", tc.name)
		}
	}
}

func TestCheckBaseURL(t *testing.T) {
	for _, ok := range []string{"https://mirror.example/reflop", "http://127.0.0.1:8080/r", "http://localhost/r", "http://[::1]:9/r"} {
		if err := CheckBaseURL(ok); err != nil {
			t.Errorf("%s rejected: %v", ok, err)
		}
	}
	for _, bad := range []string{"http://mirror.example/reflop", "ftp://x/y", "file:///tmp/r", "mirror.example", "http://127.0.0.1.evil.example/"} {
		if err := CheckBaseURL(bad); err == nil {
			t.Errorf("%s accepted", bad)
		}
	}
}

// The release keys are embedded three times; they must agree.
func TestReleaseKeysInSync(t *testing.T) {
	sh, err := os.ReadFile("../../../../install.sh")
	if err != nil {
		t.Fatal(err)
	}
	py, err := os.ReadFile("../../../../client/python/reflop/update.py")
	if err != nil {
		t.Fatal(err)
	}
	for _, k := range ReleaseKeys {
		f := strings.Fields(k)
		key := f[0] + " " + f[1]
		if !bytes.Contains(sh, []byte(key)) {
			t.Errorf("install.sh does not embed release key %s", key)
		}
		if !bytes.Contains(py, []byte(key)) {
			t.Errorf("client/python/reflop/update.py does not embed release key %s", key)
		}
	}
	if n := bytes.Count(sh, []byte("ssh-ed25519 ")); n != len(ReleaseKeys) {
		t.Errorf("install.sh embeds %d keys, reflopd %d", n, len(ReleaseKeys))
	}
	if n := bytes.Count(py, []byte(`"ssh-ed25519 `)); n != len(ReleaseKeys) {
		t.Errorf("update.py embeds %d keys, reflopd %d", n, len(ReleaseKeys))
	}
}
