//go:build !unix

package tools

import "os/exec"

// Process groups are a unix notion; elsewhere the timeout kills bash
// alone and WaitDelay still bounds the wait.
func ownProcessGroup(cmd *exec.Cmd) {}

func killProcessGroup(cmd *exec.Cmd) {}
