import type { ButtonHTMLAttributes, ReactNode } from "react";

type Props = ButtonHTMLAttributes<HTMLButtonElement> & {
  icon?: ReactNode;
  variant?: "primary" | "secondary" | "danger" | "disabled";
};

export function Button({ icon, children, variant = "secondary", className = "", ...props }: Props) {
  const styles = {
    primary: "bg-zero-accent text-white hover:bg-[#95c55d]",
    secondary: "bg-zero-panel2 text-zinc-100 hover:bg-[#3c3935]",
    danger: "bg-[#7a302f] text-white hover:bg-[#8e3837]",
    disabled: "bg-[#34312d] text-zinc-500 cursor-not-allowed"
  };
  return (
    <button
      className={`inline-flex items-center justify-center gap-2 rounded-md px-4 py-3 text-sm font-semibold transition-colors disabled:cursor-not-allowed disabled:bg-[#34312d] disabled:text-zinc-500 ${styles[variant]} ${className}`}
      {...props}
    >
      {icon}
      {children}
    </button>
  );
}
