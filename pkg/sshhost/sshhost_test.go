// SPDX-License-Identifier: Apache-2.0

package sshhost

import (
	"os"
	"path/filepath"
	"testing"
)

const (
	ed25519Key = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIEUt5SeoPVVwmna62vaHwvgGplV4vXWKREhOZHA4/jFh"
	ecdsaKey   = "ecdsa-sha2-nistp256 AAAAE2VjZHNhLXNoYTItbmlzdHAyNTYAAAAIbmlzdHAyNTYAAABBBKFaml54TqFwdHwDyok9nG50rhstSIzW9I2khyCatx52lwwS9eK2NMoQN0jaIxXT7vLac24ZPwbAqJeSusBfXZ4="
	rsaKey     = "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQC6LB543nAKtSOyqqjcJpVoICvy9zC6Bp8xhlN8aX8d0+JB75BULSdhXyFwDIZzO8FamFC5W/+PQ08yeu4LglBrHx2WGC70kKqaIu4AzKHR2562O3xwHWmmfmK45UwAC+P2xuACJDwbwNXGVdkqCgLbs+0H+YWssiWm5/xDEriq+1rdBL4rfP9xHEQbJGxOwso0OGXxTL76WWk9HQdO6FRwQA+O9LV792kHEdU0MsSvY8yEpCCkVKExUlqKjrROZooYhwmYrHCicx9g5+8rIsgnwfs+QB5RxdV9Cj+fes7Bvu75G22ak2/GHPxuod5sVcimIn9O/yNyzG6To/Ll31r33JSM5lPRKdZaaoUfzwh96PvFYarMLqT+0/T9VNruRY6QT1K/V0VJ73Ju5/NugTozKKorCLSQd1aeb2/GGbt6pLmDRrISXbWHXO0jIwZbyVrTlTw9O+SuZkfIsWI7c7TlJc0pQWltiWASAX9jSx0SWbI3YigVrdeK/JE8jqn5pVM="
)

func TestLocalHostKeys(t *testing.T) {
	dir := t.TempDir()
	write := func(name, line string) { os.WriteFile(filepath.Join(dir, name), []byte(line+" root@host\n"), 0o644) }
	write("ssh_host_rsa_key.pub", rsaKey)
	write("ssh_host_ecdsa_key.pub", ecdsaKey)
	write("ssh_host_ed25519_key.pub", ed25519Key)
	write("ssh_host_bogus_key.pub", "ssh-ed25519 AAAAB3NzaC1yc2EAAAADAQAB")
	got, err := LocalHostKeys(filepath.Join(dir, "ssh_host_*_key.pub"))
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 || got[0] != ed25519Key || got[1] != ecdsaKey {
		t.Fatalf("got %q; want ed25519 then ecdsa, no RSA, no malformed key", got)
	}
	os.Remove(filepath.Join(dir, "ssh_host_ecdsa_key.pub"))
	os.Remove(filepath.Join(dir, "ssh_host_ed25519_key.pub"))
	if got, _ := LocalHostKeys(filepath.Join(dir, "ssh_host_*_key.pub")); len(got) != 1 || got[0] != rsaKey {
		t.Fatalf("RSA-only host: got %q", got)
	}
}

func TestParseRejects(t *testing.T) {
	for _, bad := range []string{"", "ssh-ed25519", "ssh-dss AAAAB3NzaC1kc3M=", "ssh-ed25519 !!!", "ssh-rsa " + ed25519Key[len("ssh-ed25519 "):]} {
		if _, _, err := Parse(bad); err == nil {
			t.Errorf("Parse(%q) accepted", bad)
		}
	}
	if _, err := QuoteOption("a\"b"); err == nil {
		t.Error("quote in path accepted")
	}
}
