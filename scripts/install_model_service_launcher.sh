#!/usr/bin/env bash

set -euo pipefail

MARKER="# model-service auto-venv launcher"
PATH_MARKER="# model-service user launcher path"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${MODEL_SERVICE_PROJECT_ROOT:-$(dirname "$SCRIPT_DIR")}"
LAUNCHER_BIN_DIR="${MODEL_SERVICE_LAUNCHER_BIN_DIR:-${HOME}/.local/bin}"
LAUNCHER_PATH="${MODEL_SERVICE_LAUNCHER_PATH:-${LAUNCHER_BIN_DIR}/model-service}"
PROFILE_PATH="${MODEL_SERVICE_PROFILE_PATH:-${HOME}/.profile}"
FORCE_INSTALL="${MODEL_SERVICE_LAUNCHER_FORCE:-0}"
UPDATE_PROFILE="${MODEL_SERVICE_LAUNCHER_UPDATE_PROFILE:-1}"

print_ok() {
    echo "OK $1"
}

print_warn() {
    echo "WARN $1"
}

print_err() {
    echo "ERROR $1" >&2
}

write_launcher() {
    local tmp_path quoted_project_root

    mkdir -p "${LAUNCHER_BIN_DIR}"
    quoted_project_root="$(printf '%q' "${PROJECT_ROOT}")"
    tmp_path="$(mktemp "${LAUNCHER_BIN_DIR}/.model-service.XXXXXX")"

    cat > "${tmp_path}" <<LAUNCHER
#!/usr/bin/env bash
${MARKER}
set -euo pipefail

PROJECT_ROOT=${quoted_project_root}
if [[ -n "\${MODEL_SERVICE_PROJECT_ROOT:-}" ]]; then
    PROJECT_ROOT="\${MODEL_SERVICE_PROJECT_ROOT}"
fi

VENV_PATH="\${PROJECT_ROOT}/.venv"
ACTIVATE_PATH="\${VENV_PATH}/bin/activate"
ENTRYPOINT="\${VENV_PATH}/bin/model-service"

if [[ ! -f "\${ACTIVATE_PATH}" ]]; then
    echo "model-service launcher: missing \${ACTIVATE_PATH}" >&2
    echo "Run: cd \"\${PROJECT_ROOT}\" && ./scripts/setup_jetson.sh" >&2
    exit 127
fi

# shellcheck disable=SC1090
source "\${ACTIVATE_PATH}"
cd "\${PROJECT_ROOT}"

if [[ ! -x "\${ENTRYPOINT}" ]]; then
    echo "model-service launcher: missing executable \${ENTRYPOINT}" >&2
    echo "Run: cd \"\${PROJECT_ROOT}\" && ./scripts/setup_jetson.sh" >&2
    exit 127
fi

exec "\${ENTRYPOINT}" "\$@"
LAUNCHER

    chmod 0755 "${tmp_path}"
    mv "${tmp_path}" "${LAUNCHER_PATH}"
}

install_launcher() {
    if [[ -e "${LAUNCHER_PATH}" ]] && ! grep -Fq "${MARKER}" "${LAUNCHER_PATH}" 2>/dev/null; then
        if [[ "${FORCE_INSTALL}" != "1" ]]; then
            print_err "Refusing to overwrite existing non-model-service launcher: ${LAUNCHER_PATH}"
            print_err "Set MODEL_SERVICE_LAUNCHER_FORCE=1 to replace it."
            exit 1
        fi
        print_warn "Replacing existing non-model-service launcher because MODEL_SERVICE_LAUNCHER_FORCE=1"
    fi

    write_launcher
    print_ok "Installed model-service auto-venv launcher: ${LAUNCHER_PATH}"
}

ensure_profile_path() {
    local profile_bin_dir

    case ":${PATH:-}:" in
        *:"${LAUNCHER_BIN_DIR}":*)
            print_ok "${LAUNCHER_BIN_DIR} is already in PATH"
            return
            ;;
    esac

    if [[ "${UPDATE_PROFILE}" != "1" ]]; then
        print_warn "${LAUNCHER_BIN_DIR} is not in PATH; profile update disabled"
        return
    fi

    mkdir -p "$(dirname "${PROFILE_PATH}")"
    touch "${PROFILE_PATH}"

    if grep -Fq "${PATH_MARKER}" "${PROFILE_PATH}"; then
        print_warn "${LAUNCHER_BIN_DIR} is not in current PATH; restart the shell or run:"
        echo "  export PATH=\"${LAUNCHER_BIN_DIR}:\$PATH\""
        return
    fi

    profile_bin_dir="$(printf '%q' "${LAUNCHER_BIN_DIR}")"
    cat >> "${PROFILE_PATH}" <<PROFILE

${PATH_MARKER}
_model_service_launcher_bin=${profile_bin_dir}
if [ -d "\$_model_service_launcher_bin" ]; then
    case ":\$PATH:" in
        *:"\$_model_service_launcher_bin":*) ;;
        *) export PATH="\$_model_service_launcher_bin:\$PATH" ;;
    esac
fi
unset _model_service_launcher_bin
PROFILE

    print_ok "Added model-service launcher PATH hook to ${PROFILE_PATH}"
    print_warn "Open a new terminal or run:"
    echo "  export PATH=\"${LAUNCHER_BIN_DIR}:\$PATH\""
}

install_launcher
ensure_profile_path
