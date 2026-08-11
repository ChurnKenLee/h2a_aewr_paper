{ pkgs, ... }:

{
  packages = with pkgs; [
    coreutils
    curl
    git
    gnugrep
    gnumake
    gnused
    jq
    actionlint
    shellcheck
    yq-go
    zola
  ];

  scripts.agent-grounding = {
    description = "Inspect, verify, generate, or query coding-agent grounding";
    exec = ''
      exec python scripts/agent_grounding.py "$@"
    '';
  };

  scripts.agent-docs-generate = {
    description = "Regenerate machine-derived Zola agent documentation";
    exec = ''
      exec python scripts/agent_grounding.py generate
    '';
  };

  scripts.agent-docs-check = {
    description = "Verify grounding freshness and build the Zola site";
    exec = ''
      python scripts/agent_grounding.py verify
      exec zola --root agent-docs check --skip-external-links
    '';
  };

  scripts.agent-docs-serve = {
    description = "Serve the local coding-agent knowledge base";
    exec = ''
      python scripts/agent_grounding.py verify
      exec zola --root agent-docs serve "$@"
    '';
  };
}
