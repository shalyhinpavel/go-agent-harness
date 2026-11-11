//go:build unix

package tools

import (
	"os/exec"
	"syscall"
)

// ownProcessGroup puts the command in its own process group, and makes a
// timeout kill the group rather than bash alone — children die with it.
func ownProcessGroup(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	cmd.Cancel = func() error { return syscall.Kill(-cmd.Process.Pid, syscall.SIGKILL) }
}

// killProcessGroup ends whatever the command left running in its group.
func killProcessGroup(cmd *exec.Cmd) {
	if cmd.Process != nil {
		_ = syscall.Kill(-cmd.Process.Pid, syscall.SIGKILL)
	}
}
