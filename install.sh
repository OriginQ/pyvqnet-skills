#!/bin/bash
# install.sh - VQNet 2.18.1 Quantum Machine Learning Skill installer
# 将本技能安装到支持的 AI 工具技能目录（OpenCode / Claude Code / Gemini CLI / Codex / Cline）

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SKILL_NAME="vqnet2-api"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

TOOLS=(opencode claude gemini codex cline)

# 复制技能文件（排除开发产物）
copy_skill() {
    local dest_dir="$1"
    local dest="$dest_dir/$SKILL_NAME"

    mkdir -p "$dest_dir"
    rm -rf "$dest"
    mkdir -p "$dest"

    tar -C "$SCRIPT_DIR" \
        --exclude='./.git' \
        --exclude='./.omo' \
        --exclude='./install.sh' \
        --exclude='./UPDATE_PLAN_v2.18.1.md' \
        -cf - . | tar -C "$dest" -xf -

    echo -e "${GREEN}✅ 已安装:${NC} $dest"
}

# 安装到指定工具; $1=工具名, $2=global|project
install_tool() {
    local tool="$1"
    local scope="${2:-global}"
    local dir=""

    # Cline 仅支持全局
    if [ "$tool" = "cline" ] && [ "$scope" = "project" ]; then
        echo -e "${YELLOW}Cline 仅支持全局安装，已跳过项目模式。${NC}"
        return 0
    fi

    if [ "$scope" = "project" ]; then
        dir="$PWD/.$tool/skills"
    else
        case "$tool" in
            opencode) dir="$HOME/.config/opencode/skills" ;;
            claude)   dir="$HOME/.claude/skills" ;;
            gemini)   dir="$HOME/.gemini/skills" ;;
            codex)    dir="$HOME/.codex/skills" ;;
            cline)    dir="$HOME/.cline/skills" ;;
            *) echo -e "${YELLOW}未知工具: $tool${NC}"; return 1 ;;
        esac
    fi

    copy_skill "$dir"
}

show_usage() {
    echo ""
    echo "用法: $0 [工具] [--project]"
    echo ""
    echo "工具（可多选）:"
    echo "  --opencode    OpenCode"
    echo "  --claude      Claude Code"
    echo "  --gemini      Gemini CLI"
    echo "  --codex       Codex"
    echo "  --cline       Cline（仅全局）"
    echo "  --all         以上全部工具"
    echo ""
    echo "选项:"
    echo "  --project     安装到当前项目的技能目录（默认安装到用户/全局目录）"
    echo "  --help        显示本帮助"
    echo ""
    echo "示例:"
    echo "  $0 --claude                    # 安装到 Claude Code 全局技能目录"
    echo "  $0 --opencode --project        # 安装到当前项目的 OpenCode 技能目录"
    echo "  $0 --all                       # 安装到全部支持工具（全局）"
    echo ""
}

main() {
    local tools=()
    local scope="global"

    for arg in "$@"; do
        case "$arg" in
            --opencode) tools+=("opencode") ;;
            --claude)   tools+=("claude") ;;
            --gemini)   tools+=("gemini") ;;
            --codex)    tools+=("codex") ;;
            --cline)    tools+=("cline") ;;
            --all)      tools=("${TOOLS[@]}") ;;
            --project)  scope="project" ;;
            --help|-h)  show_usage; exit 0 ;;
            *) echo -e "${YELLOW}未知参数: $arg${NC}"; show_usage; exit 1 ;;
        esac
    done

    # 无参数时进入交互菜单
    if [ ${#tools[@]} -eq 0 ]; then
        echo -e "${BLUE}选择要安装到的工具:${NC}"
        echo "  1) OpenCode"
        echo "  2) Claude Code"
        echo "  3) Gemini CLI"
        echo "  4) Codex"
        echo "  5) Cline"
        echo "  6) 全部（全局）"
        echo ""
        if [ ! -t 0 ]; then
            show_usage
            exit 1
        fi
        read -p "请选择 [1-6]: " choice
        case "$choice" in
            1) tools=("opencode") ;;
            2) tools=("claude") ;;
            3) tools=("gemini") ;;
            4) tools=("codex") ;;
            5) tools=("cline") ;;
            6) tools=("${TOOLS[@]}") ;;
            *) echo -e "${YELLOW}无效选择。${NC}"; exit 1 ;;
        esac
    fi

    echo -e "${BLUE}VQNet 2.18.1 Skill 安装（${scope}）${NC}"
    for tool in "${tools[@]}"; do
        install_tool "$tool" "$scope"
    done

    echo ""
    echo -e "${GREEN}安装完成。重启对应 AI 工具后即可通过 VQNet/pyvqnet 等关键词触发本技能。${NC}"
}

main "$@"
