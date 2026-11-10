// SPDX-License-Identifier: Apache-2.0

// Package auth manages the server key, Guest identities, invites, and the
// challenge-response authentication of PLAN.md §15 (ADR-006).
package auth

import (
	"crypto/ed25519"
	"crypto/rand"
	"crypto/sha256"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"errors"
	"fmt"
	"math/big"
	"os"
	"time"

	"github.com/chaseungjoon/reflop/server/reflopd/internal/config"
)

// ServerKey is reflopd's long-term identity.
type ServerKey struct {
	Private     ed25519.PrivateKey
	Cert        tls.Certificate
	Fingerprint [32]byte // SHA-256 of the SubjectPublicKeyInfo
}

// LoadOrCreateServerKey loads the server key, creating it on first run.
func LoadOrCreateServerKey(keyPath, certPath string) (*ServerKey, error) {
	if _, err := os.Stat(keyPath); errors.Is(err, os.ErrNotExist) {
		if err := createServerKey(keyPath, certPath); err != nil {
			return nil, err
		}
	}
	keyPEM, err := os.ReadFile(keyPath)
	if err != nil {
		return nil, err
	}
	certPEM, err := os.ReadFile(certPath)
	if err != nil {
		return nil, err
	}
	if st, err := os.Stat(keyPath); err == nil && st.Mode().Perm()&0o077 != 0 {
		return nil, fmt.Errorf("%s is accessible by other users (mode %v); run: chmod 600 %s", keyPath, st.Mode().Perm(), keyPath)
	}
	cert, err := tls.X509KeyPair(certPEM, keyPEM)
	if err != nil {
		return nil, fmt.Errorf("server key: %w", err)
	}
	priv, ok := cert.PrivateKey.(ed25519.PrivateKey)
	if !ok {
		return nil, errors.New("server key is not Ed25519")
	}
	leaf, err := x509.ParseCertificate(cert.Certificate[0])
	if err != nil {
		return nil, err
	}
	cert.Leaf = leaf
	return &ServerKey{Private: priv, Cert: cert, Fingerprint: SPKIFingerprint(leaf)}, nil
}

// SPKIFingerprint is SHA-256 over the certificate's SubjectPublicKeyInfo.
func SPKIFingerprint(c *x509.Certificate) [32]byte { return sha256.Sum256(c.RawSubjectPublicKeyInfo) }

func createServerKey(keyPath, certPath string) error {
	pub, priv, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		return err
	}
	serial, err := rand.Int(rand.Reader, new(big.Int).Lsh(big.NewInt(1), 127))
	if err != nil {
		return err
	}
	tmpl := &x509.Certificate{
		SerialNumber:          serial,
		Subject:               pkix.Name{CommonName: "reflopd"},
		NotBefore:             time.Now().Add(-time.Hour),
		NotAfter:              time.Now().AddDate(100, 0, 0),
		KeyUsage:              x509.KeyUsageDigitalSignature | x509.KeyUsageCertSign,
		ExtKeyUsage:           []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
		BasicConstraintsValid: true,
		IsCA:                  true,
		DNSNames:              []string{"reflopd"},
	}
	der, err := x509.CreateCertificate(rand.Reader, tmpl, tmpl, pub, priv)
	if err != nil {
		return err
	}
	pk, err := x509.MarshalPKCS8PrivateKey(priv)
	if err != nil {
		return err
	}
	if err := config.WriteFileAtomic(keyPath, pem.EncodeToMemory(&pem.Block{Type: "PRIVATE KEY", Bytes: pk}), 0o600); err != nil {
		return err
	}
	return config.WriteFileAtomic(certPath, pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: der}), 0o644)
}

// Domain separation prefixes for signed and MACed messages.
const (
	AuthContext = "reflop-auth-v1\x00"
	PairContext = "reflop-pair-v1\x00"
)

// AuthMessage is what a client signs to authenticate.
func AuthMessage(nonce, serverFP, pub []byte) []byte {
	m := make([]byte, 0, len(AuthContext)+len(nonce)+len(serverFP)+len(pub))
	m = append(m, AuthContext...)
	m = append(m, nonce...)
	m = append(m, serverFP...)
	return append(m, pub...)
}

// PairMessage is what a client MACs and signs to pair.
func PairMessage(nonce, serverFP, inviteID, pub []byte) []byte {
	m := make([]byte, 0, len(PairContext)+len(nonce)+len(serverFP)+len(inviteID)+len(pub))
	m = append(m, PairContext...)
	m = append(m, nonce...)
	m = append(m, serverFP...)
	m = append(m, inviteID...)
	return append(m, pub...)
}
