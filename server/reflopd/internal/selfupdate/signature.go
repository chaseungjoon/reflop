// SPDX-License-Identifier: Apache-2.0

package selfupdate

import (
	"bytes"
	"crypto/sha256"
	"crypto/sha512"
	"encoding/pem"
	"errors"
	"fmt"

	"golang.org/x/crypto/ssh"
)

// ReleaseKeys are the public keys that sign releases. Each release's
// checksums.txt carries a detached OpenSSH signature, checksums.txt.sig,
// made offline with `ssh-keygen -Y sign -n reflop-release` (SEC_SWEEP M4).
// install.sh and the Python client embed the same list; a test keeps the
// copies in sync. To rotate, add the new key, release, then drop the old.
var ReleaseKeys = []string{
	"ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIKqs9C0dx4nGfqCL1Vw0hgwleOGNaDOPYyMBCwYU3lsW reflop-release",
}

// SignatureNamespace binds signatures to reflop releases, so a signature
// made with the same key for another purpose never verifies here.
const SignatureNamespace = "reflop-release"

// ErrSignature means a release's signature is missing or invalid.
var ErrSignature = errors.New("release signature verification failed")

// VerifySignature checks an armored OpenSSH signature (the SSHSIG format of
// ssh-keygen -Y sign) over msg against keys.
func VerifySignature(msg, armored []byte, keys []string) error {
	blk, _ := pem.Decode(bytes.TrimSpace(armored))
	if blk == nil || blk.Type != "SSH SIGNATURE" {
		return fmt.Errorf("%w: not an SSH signature", ErrSignature)
	}
	var sig struct {
		Magic     [6]byte
		Version   uint32
		PublicKey []byte
		Namespace string
		Reserved  string
		HashAlg   string
		Signature []byte
	}
	if err := ssh.Unmarshal(blk.Bytes, &sig); err != nil || string(sig.Magic[:]) != "SSHSIG" || sig.Version != 1 {
		return fmt.Errorf("%w: malformed signature", ErrSignature)
	}
	if sig.Namespace != SignatureNamespace {
		return fmt.Errorf("%w: signature is for %q, not %q", ErrSignature, sig.Namespace, SignatureNamespace)
	}
	trusted := false
	for _, k := range keys {
		pk, _, _, _, err := ssh.ParseAuthorizedKey([]byte(k))
		if err == nil && bytes.Equal(pk.Marshal(), sig.PublicKey) {
			trusted = true
		}
	}
	if !trusted {
		return fmt.Errorf("%w: signed by a key that is not a reflop release key", ErrSignature)
	}
	pub, err := ssh.ParsePublicKey(sig.PublicKey)
	if err != nil {
		return fmt.Errorf("%w: %v", ErrSignature, err)
	}
	var h []byte
	switch sig.HashAlg {
	case "sha512":
		s := sha512.Sum512(msg)
		h = s[:]
	case "sha256":
		s := sha256.Sum256(msg)
		h = s[:]
	default:
		return fmt.Errorf("%w: unsupported hash %q", ErrSignature, sig.HashAlg)
	}
	signed := ssh.Marshal(struct {
		Namespace string
		Reserved  string
		HashAlg   string
		Hash      []byte
	}{sig.Namespace, sig.Reserved, sig.HashAlg, h})
	signed = append([]byte("SSHSIG"), signed...)
	var s ssh.Signature
	if err := ssh.Unmarshal(sig.Signature, &s); err != nil {
		return fmt.Errorf("%w: malformed signature", ErrSignature)
	}
	if s.Format != pub.Type() {
		return fmt.Errorf("%w: signature algorithm %q does not match the key", ErrSignature, s.Format)
	}
	if err := pub.Verify(signed, &s); err != nil {
		return fmt.Errorf("%w: bad signature", ErrSignature)
	}
	return nil
}
